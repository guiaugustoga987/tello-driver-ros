#!/usr/bin/env python2
import rospy
from std_msgs.msg import Empty, UInt8, Bool
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from dynamic_reconfigure.server import Server
from h264_image_transport.msg import H264Packet
from tello_driver.msg import TelloStatus
from tello_driver.cfg import TelloConfig
from cv_bridge import CvBridge, CvBridgeError
from rospkg import RosPack
import camera_info_manager as cim

import av
import math
import numpy as np
import threading
import time
from tellopy._internal import tello
from tellopy._internal import error
from tellopy._internal import protocol
from tellopy._internal import logger

def euler_to_quaternion(pitch, roll, yaw):
        pitch = math.radians(pitch)
        roll = math.radians(roll)
        yaw = math.radians(yaw)
        
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qw, qx, qy, qz]

class RospyLogger(logger.Logger):
    def __init__(self, header=''):
        super(RospyLogger, self).__init__(header)

    def error(self, s):
        if self.log_level < logger.LOG_ERROR:
            return
        rospy.logerr(s)

    def warn(self, s):
        if self.log_level < logger.LOG_WARN:
            return
        rospy.logwarn(s)

    def info(self, s):
        if self.log_level < logger.LOG_INFO:
            return
        rospy.loginfo(s)

    def debug(self, s):
        if self.log_level < logger.LOG_DEBUG:
            return
        rospy.logdebug(s)


def notify_cmd_success(cmd, success):
    if success:
        rospy.loginfo('%s command executed' % cmd)
    else:
        rospy.logwarn('%s command failed' % cmd)


class TelloNode(tello.Tello):
    def __init__(self):
        # Fetch parameters
        self.local_cmd_client_port = int(
            rospy.get_param('~local_cmd_client_port', 8890))
        self.local_vid_server_port = int(
            rospy.get_param('~local_vid_server_port', 6038))
        self.tello_ip = rospy.get_param('~tello_ip', '192.168.10.1')
        self.tello_cmd_server_port = int(
            rospy.get_param('~tello_cmd_server_port', 8889))
        self.connect_timeout_sec = float(
            rospy.get_param('~connect_timeout_sec', 10.0))
        self.stream_h264_video = bool(
            rospy.get_param('~stream_h264_video', False))
        self.tello_sdk = bool(
            rospy.get_param('~tello_sdk', True))            
        self.bridge = CvBridge()
        self.frame_thread = None
        self.cam_fps = rospy.get_param('~cam_fps', 40)      
        default_calib_path = RosPack().get_path('tello_driver') + '/cam_calib/default.yaml'
        self.calib_path = rospy.get_param('~camera_calib', default_calib_path) 
        self.caminfo = cim.loadCalibrationFile(self.calib_path, 'camera_front')
        self.caminfo.header.frame_id = rospy.get_param('~camera_frame', 'camera_front')        

        # Connect to drone
        log = RospyLogger('Tello')
        log.set_level(self.LOG_INFO)
        super(TelloNode, self).__init__(
            local_cmd_client_port=self.local_cmd_client_port,
            local_vid_server_port=self.local_vid_server_port,
            tello_ip=self.tello_ip,
            tello_cmd_server_port=self.tello_cmd_server_port,
            tello_sdk = self.tello_sdk,
            log=log)
        rospy.loginfo('Connecting to drone @ %s:%d' % self.tello_addr)
        self.connect()
        try:
            self.wait_for_connection(timeout=self.connect_timeout_sec)
        except error.TelloError as err:
            rospy.logerr(str(err))
            rospy.signal_shutdown(str(err))
            self.quit()
            return
        rospy.loginfo('Connected to drone')
        rospy.on_shutdown(self.cb_shutdown)

        # Setup dynamic reconfigure
        self.cfg = None
        self.srv_dyncfg = Server(TelloConfig, self.cb_dyncfg)

        # Setup topics and services
        # NOTE: ROS interface deliberately made to resemble bebop_autonomy
        self.pub_status = rospy.Publisher(
            'status', TelloStatus, queue_size=1, latch=True)
        self.pub_odom = rospy.Publisher(
            'odom', Odometry, queue_size=1, latch=True)            
        self.pub_caminfo = rospy.Publisher('camera/camera_info', CameraInfo, queue_size=1, latch=True)
        if self.stream_h264_video:
            self.pub_image_h264 = rospy.Publisher(
                'camera/image_raw/h264', H264Packet, queue_size=10)
        else:
            self.pub_image_raw = rospy.Publisher(
                'camera/image_raw', Image, queue_size=10)                
        self.sub_ap_connect = rospy.Subscriber('ap_connect', Empty, self.cb_ap_connect)           
        self.sub_videomode = rospy.Subscriber('toggle_cam', Empty, self.cb_videomode)           
        self.sub_takeoff = rospy.Subscriber('takeoff', Empty, self.cb_takeoff)
        self.sub_throw_takeoff = rospy.Subscriber(
            'auto_takeoff', Empty, self.cb_throw_takeoff)
        self.sub_land = rospy.Subscriber('land', Empty, self.cb_land)
# TODO        self.sub_palm_land = rospy.Subscriber('palm_land', Empty, self.cb_palm_land)
        self.sub_emergency = rospy.Subscriber('emergency', Empty, self.cb_emergency, queue_size=1)    
        self.sub_flattrim = rospy.Subscriber(
            'flattrim', Empty, self.cb_flattrim)
        self.sub_flip = rospy.Subscriber('flip', UInt8, self.cb_flip)
        self.sub_cmd_vel = rospy.Subscriber('cmd_vel', Twist, self.cb_cmd_vel, queue_size=1)
        self.sub_fast_mode = rospy.Subscriber('pilot_mode', Empty, self.cb_fast_mode)

        self.subscribe(self.EVENT_FLIGHT_DATA, self.cb_status_telem)
        self.subscribe(self.EVENT_LOG_DATA, self.cb_status_log)

        if self.stream_h264_video:
            self.start_video()
            self.subscribe(self.EVENT_VIDEO_FRAME, self.cb_h264_frame)
        else:
            if self.tello_sdk:
                self.stream_on()
                self.subscribe(self.EVENT_VIDEO_FRAME, self.cb_frame)
            else:
                self.frame_thread = threading.Thread(target=self.framegrabber_loop)
                self.frame_thread.start()            

        # NOTE: odometry from parsing logs might be possible eventually,
        #       but it is unclear from tests what's being sent by Tello
        # - https://github.com/Kragrathea/TelloLib/blob/master/TelloLib/Tello.cs#L1047
        # - https://github.com/Kragrathea/TelloLib/blob/master/TelloLib/parsedRecSpecs.json
        # self.pub_odom = rospy.Publisher(
        #    'odom', UInt8MultiArray, queue_size=1, latch=True)
        # self.pub_odom = rospy.Publisher(
        #    'odom', Odometry, queue_size=1, latch=True)
        #self.subscribe(self.EVENT_LOG, self.cb_odom_log)

        rospy.loginfo('Tello driver node ready')

    def cb_shutdown(self):
        self.quit()
        if self.frame_thread is not None:
            self.frame_thread.join()

    def cb_status_log(self, event, sender, data, **args):
        if self.tello_sdk:
            data.imu.q0, data.imu.q1, data.imu.q2, data.imu.q3 = euler_to_quaternion(data.imu.gyro_x, data.imu.gyro_y, data.imu.gyro_z)

        ### Odometry message
        msg = Odometry()
        ### Height measured in meters
        msg.pose.pose.position.z = -data.mvo.pos_z*1000
        msg.pose.pose.position.x = data.mvo.pos_x
        msg.pose.pose.position.y = data.mvo.pos_y
        msg.pose.pose.orientation.w = data.imu.q0
        msg.pose.pose.orientation.x = data.imu.q1
        msg.pose.pose.orientation.y = data.imu.q2
        msg.pose.pose.orientation.z = data.imu.q3
        ### Speed in m/sec
        msg.twist.twist.linear.x = data.mvo.vel_y
        msg.twist.twist.linear.y = data.mvo.vel_x
        msg.twist.twist.linear.z = -data.mvo.vel_z                                 
        
        msg.child_frame_id = 'Tello'
        msg.header.stamp = rospy.Time.now()
        
        self.pub_odom.publish(msg)
#        print('record_log: %s: %s' % (event.name, str(data)))    

    def cb_status_telem(self, event, sender, data, **args):
        speed_horizontal_mps = math.sqrt(
            data.north_speed*data.north_speed+data.east_speed*data.east_speed)/10.

        # TODO: verify outdoors: anecdotally, observed that:
        # data.east_speed points to South
        # data.north_speed points to East
        msg = TelloStatus(
            height_m=data.height/10.,
            speed_northing_mps=-data.east_speed/10.,
            speed_easting_mps=data.north_speed/10.,
            speed_horizontal_mps=speed_horizontal_mps,
# CHECK            speed_vertical_mps=-data.vertical_speed/10.,
            flight_time_sec=data.fly_time/10.,
            imu_state=data.imu_state,
            pressure_state=data.pressure_state,
            down_visual_state=data.down_visual_state,
            power_state=data.power_state,
            battery_state=data.battery_state,
            gravity_state=data.gravity_state,
            wind_state=data.wind_state,
            imu_calibration_state=data.imu_calibration_state,
            battery_percentage=data.battery_percentage,
            drone_fly_time_left_sec=data.drone_fly_time_left/10.,
            drone_battery_left_sec=data.drone_battery_left/10.,
            is_flying=data.em_sky,
            is_on_ground=data.em_ground,
            is_em_open=data.em_open,
            is_drone_hover=data.drone_hover,
            is_outage_recording=data.outage_recording,
            is_battery_low=data.battery_low,
            is_battery_lower=data.battery_lower,
            is_factory_mode=data.factory_mode,
            fly_mode=data.fly_mode,
            throw_takeoff_timer_sec=data.throw_fly_timer/10.,
            camera_state=data.camera_state,
            electrical_machinery_state=data.electrical_machinery_state,
            front_in=data.front_in,
            front_out=data.front_out,
            front_lsc=data.front_lsc,
            temperature_height_m=data.temperature_height/10.,
            cmd_roll_ratio=self.right_x,
            cmd_pitch_ratio=self.right_y,
            cmd_yaw_ratio=self.left_x,
            cmd_vspeed_ratio=self.left_y,
            cmd_fast_mode=self.fast_mode,
        )
        self.pub_status.publish(msg)

    def cb_odom_log(self, event, sender, data, **args):
        odom_msg = UInt8MultiArray()
        odom_msg.data = str(data)
        self.pub_odom.publish(odom_msg)

    def cb_ap_connect(self, msg):
        ssid = 'HOTSPOT'
        password = 'qwertyuiop'
        success = self.connect_to_ap(ssid, password)
        notify_cmd_success('Connect ap', success)
    
    def cb_frame(self, event, sender, data, **args):
        stamp = rospy.Time.now()
        img_msg = self.bridge.cv2_to_imgmsg(data, 'rgb8')
        img_msg.header.frame_id = rospy.get_namespace()
        img_msg.header.stamp = stamp        
        self.pub_image_raw.publish(img_msg)
                    
        self.caminfo.header.stamp = stamp
        self.pub_caminfo.publish(self.caminfo)                    
        
    def framegrabber_loop(self):
        # Repeatedly try to connect
        vs = self.get_video_stream()
        while self.state != self.STATE_QUIT:
            try:
                container = av.open(vs)
                break
            except BaseException as err:
                rospy.logerr('fgrab: pyav stream failed - %s' % str(err))
                time.sleep(1.0)
        
        # Once connected, process frames till drone/stream closes
        while self.state != self.STATE_QUIT:
            # skip first 300 frames
            frame_skip = 300
            try:
                # vs blocks, dies on self.stop
                for frame in container.decode(video=0):
                    if 0 < frame_skip:
                        frame_skip = frame_skip -1
                        continue            
                    img = np.array(frame.to_image())
                    try:
                        stamp = rospy.Time.now()
                        img_msg = self.bridge.cv2_to_imgmsg(img, 'rgb8')
                        img_msg.header.frame_id = rospy.get_namespace()
                        img_msg.header.stamp = stamp
                    except CvBridgeError as err:
                        rospy.logerr('fgrab: cv bridge failed - %s' % str(err))
                        continue
                    self.pub_image_raw.publish(img_msg)
                    
                    self.caminfo.header.stamp = stamp
                    self.pub_caminfo.publish(self.caminfo)                    
                break
            except BaseException as err:
                rospy.logerr('fgrab: pyav decoder failed - %s' % str(err))                
            
    def cb_dyncfg(self, config, level):
        update_all = False
        req_sps_pps = False
        if self.cfg is None:
            self.cfg = config
            update_all = True

        if update_all or self.cfg.fixed_video_rate != config.fixed_video_rate:
            self.set_video_encoder_rate(config.fixed_video_rate)
            req_sps_pps = True
        if update_all or self.cfg.vel_cmd_limit != config.vel_cmd_limit:
            self.vel_cmd_limit = config.vel_cmd_limit
        if update_all or self.cfg.vel_cmd_scale != config.vel_cmd_scale:
            self.vel_cmd_scale = config.vel_cmd_scale        
        
        self.cfg = config
        return self.cfg

    def cb_takeoff(self, msg):
        success = self.takeoff()
        notify_cmd_success('Takeoff', success)

    def cb_throw_takeoff(self, msg):
        success = self.throw_and_go()
        if success:
            rospy.loginfo('Drone set to auto-takeoff when thrown')
        else:
            rospy.logwarn('ThrowTakeoff command failed')

    def cb_land(self, msg):
        success = self.land()
        notify_cmd_success('Land', success)

    def cb_palm_land(self, msg):
        success = self.palm_land()
        notify_cmd_success('PalmLand', success)

    def cb_emergency(self, msg):
        success = self.emergency()
        notify_cmd_success('Emergency', success)

    def cb_flattrim(self, msg):
        success = self.flattrim()
        notify_cmd_success('FlatTrim', success)

    def cb_flip(self, msg):
        if msg.data < 0 or msg.data > protocol.FLIP_MAX_INT:
            rospy.logwarn('Invalid flip direction: %d' % msg.data)
            return
        success = self.flip(msg.data)
        notify_cmd_success('Flip %d' % msg.data, success)

    def cb_cmd_vel(self, msg):
        self.set_pitch(msg.linear.y)
        self.set_roll(msg.linear.x)
        self.set_yaw(msg.angular.z)
        self.set_throttle(msg.linear.z)

    def cb_fast_mode(self, msg):
        if not self.fast_mode:
            self.set_fast_mode(True)
        else:
            self.set_fast_mode(False)

    def cb_videomode(self, msg):
        if not self.zoom:
            self.set_video_mode(True)
        else:
            self.set_video_mode(False)

def main():
    rospy.init_node('tello_node')
    robot = TelloNode()

    if robot.state != robot.STATE_QUIT:
        rospy.spin()


if __name__ == '__main__':
    main()
