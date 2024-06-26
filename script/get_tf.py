#!/usr/bin/env python3

import rospy
import tf
import numpy as np

# Write a file such that by putting the box under the camera, it can automatically determine the pose shift of each marker
# Probably need to publish the marker's frame first
def average_quaternions(quaternions):
    # Convert list of quaternions to a numpy array
    q = np.array(quaternions)
    # Compute the average quaternion using the mean of each component
    q_mean = np.mean(q, axis=0)
    # Normalize the quaternion to ensure it represents a valid rotation
    q_mean /= np.linalg.norm(q_mean)
    return q_mean

def markerTF():
    rospy.init_node('marker_tf_extraction')

    listener = tf.TransformListener()

    rate = rospy.Rate(10.0)
    quaternions = []
    
    left = True
    while not rospy.is_shutdown():
        try:
            marker_id = 3
            side = 'left' if left else 'right'
            # Get the transform from /aruco_marker_10 to /base_link
            (trans, rot) = listener.lookupTransform(f'/arucomarker_{marker_id}_{side}', f'/arucomarker_10_{side}', rospy.Time(0))

            print("Rotation (quaternion): ", rot)
            quaternions.append(rot)
            
            if len(quaternions) >= 10:
                q_mean = average_quaternions(quaternions)
                np.savetxt(f'/home/riot/kinova_gen3_lite/src/box_detection/cam_poses/arucomarker_{marker_id}_left.txt', q_mean, delimiter=' ')
                rospy.signal_shutdown("frame_saved")
                
                
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        rate.sleep()

def boxFrameAdjustment():
    '''
    put the box as the reference pose to extract the pose shift matrix
    '''
    rospy.init_node('tf_echo_listener')

    listener = tf.TransformListener()

    rate = rospy.Rate(10.0)
    quaternions = []
    while not rospy.is_shutdown():
        try:
            i = 0
            # Get the transform from /aruco_marker_10 to /base_link
            (trans, rot) = listener.lookupTransform('/box_frame', '/base_link', rospy.Time(0))

            print("Rotation (quaternion): ", rot)
            quaternions.append(rot)
            
            if len(quaternions) >= 10:
                q_mean = average_quaternions(quaternions)
                np.savetxt('/home/riot/kinova_gen3_lite/src/box_detection/cam_poses/box_frame_shift.txt', q_mean, delimiter=' ')
                rospy.signal_shutdown("frame_saved")
                
                
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        rate.sleep()

if __name__ == '__main__':
    try:
        markerTF()
    except rospy.ROSInterruptException:
        pass




