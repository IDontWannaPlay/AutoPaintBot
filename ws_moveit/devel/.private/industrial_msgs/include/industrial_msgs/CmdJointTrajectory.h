// Generated by gencpp from file industrial_msgs/CmdJointTrajectory.msg
// DO NOT EDIT!


#ifndef INDUSTRIAL_MSGS_MESSAGE_CMDJOINTTRAJECTORY_H
#define INDUSTRIAL_MSGS_MESSAGE_CMDJOINTTRAJECTORY_H

#include <ros/service_traits.h>


#include <industrial_msgs/CmdJointTrajectoryRequest.h>
#include <industrial_msgs/CmdJointTrajectoryResponse.h>


namespace industrial_msgs
{

struct CmdJointTrajectory
{

typedef CmdJointTrajectoryRequest Request;
typedef CmdJointTrajectoryResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct CmdJointTrajectory
} // namespace industrial_msgs


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::industrial_msgs::CmdJointTrajectory > {
  static const char* value()
  {
    return "94fdf82abbbb1071bc31be1a2aea4fcd";
  }

  static const char* value(const ::industrial_msgs::CmdJointTrajectory&) { return value(); }
};

template<>
struct DataType< ::industrial_msgs::CmdJointTrajectory > {
  static const char* value()
  {
    return "industrial_msgs/CmdJointTrajectory";
  }

  static const char* value(const ::industrial_msgs::CmdJointTrajectory&) { return value(); }
};


// service_traits::MD5Sum< ::industrial_msgs::CmdJointTrajectoryRequest> should match
// service_traits::MD5Sum< ::industrial_msgs::CmdJointTrajectory >
template<>
struct MD5Sum< ::industrial_msgs::CmdJointTrajectoryRequest>
{
  static const char* value()
  {
    return MD5Sum< ::industrial_msgs::CmdJointTrajectory >::value();
  }
  static const char* value(const ::industrial_msgs::CmdJointTrajectoryRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::industrial_msgs::CmdJointTrajectoryRequest> should match
// service_traits::DataType< ::industrial_msgs::CmdJointTrajectory >
template<>
struct DataType< ::industrial_msgs::CmdJointTrajectoryRequest>
{
  static const char* value()
  {
    return DataType< ::industrial_msgs::CmdJointTrajectory >::value();
  }
  static const char* value(const ::industrial_msgs::CmdJointTrajectoryRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::industrial_msgs::CmdJointTrajectoryResponse> should match
// service_traits::MD5Sum< ::industrial_msgs::CmdJointTrajectory >
template<>
struct MD5Sum< ::industrial_msgs::CmdJointTrajectoryResponse>
{
  static const char* value()
  {
    return MD5Sum< ::industrial_msgs::CmdJointTrajectory >::value();
  }
  static const char* value(const ::industrial_msgs::CmdJointTrajectoryResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::industrial_msgs::CmdJointTrajectoryResponse> should match
// service_traits::DataType< ::industrial_msgs::CmdJointTrajectory >
template<>
struct DataType< ::industrial_msgs::CmdJointTrajectoryResponse>
{
  static const char* value()
  {
    return DataType< ::industrial_msgs::CmdJointTrajectory >::value();
  }
  static const char* value(const ::industrial_msgs::CmdJointTrajectoryResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // INDUSTRIAL_MSGS_MESSAGE_CMDJOINTTRAJECTORY_H