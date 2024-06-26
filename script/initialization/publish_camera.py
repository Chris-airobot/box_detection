#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
import tf.transformations as tf_trans
import numpy as np
import sys

def publish_transform():
    rospy.init_node('camera_to_robot_tf_broadcaster')

    # Retrieve parameters from the launch file
    input_file = rospy.get_param('~input_file')
    parent_frame = rospy.get_param('~parent_frame')
    child_frame = rospy.get_param('~child_frame')

    br = tf2_ros.StaticTransformBroadcaster()

    # Load the transformation matrix from the input file
    matrix = np.loadtxt(input_file, delimiter=' ')

    transform = geometry_msgs.msg.TransformStamped()
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame

    # Extract translation
    transform.transform.translation.x = matrix[0][3]
    transform.transform.translation.y = matrix[1][3]
    transform.transform.translation.z = matrix[2][3]

    # Extract rotation matrix
    rotation_matrix = [
        [matrix[0][0], matrix[0][1], matrix[0][2]],
        [matrix[1][0], matrix[1][1], matrix[1][2]],
        [matrix[2][0], matrix[2][1], matrix[2][2]]
    ]
    # tf rotation from /camera_link to /camera_depth_optical_frame
    quat_tf = [0.500, -0.500, 0.500, 0.500]
    rotation_matrix_tf = tf_trans.quaternion_matrix(quat_tf)[:3, :3]
    
    # Combine the rotations
    combined_rotation = np.dot(rotation_matrix, rotation_matrix_tf)
    # Convert combined rotation matrix to quaternion
    combined_matrix = np.eye(4)
    combined_matrix[:3, :3] = combined_rotation


    # Convert rotation matrix to quaternion
    quat = tf_trans.quaternion_from_matrix(combined_matrix)

    transform.transform.rotation.x = quat[0]
    transform.transform.rotation.y = quat[1]
    transform.transform.rotation.z = quat[2]
    transform.transform.rotation.w = quat[3]

    br.sendTransform(transform)

    rospy.spin()

if __name__ == '__main__':
    try:
        publish_transform()
    except rospy.ROSInterruptException:
        pass
