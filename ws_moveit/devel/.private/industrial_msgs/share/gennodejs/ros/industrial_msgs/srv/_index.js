
"use strict";

let SetDrivePower = require('./SetDrivePower.js')
let GetRobotInfo = require('./GetRobotInfo.js')
let SetRemoteLoggerLevel = require('./SetRemoteLoggerLevel.js')
let StartMotion = require('./StartMotion.js')
let CmdJointTrajectory = require('./CmdJointTrajectory.js')
let StopMotion = require('./StopMotion.js')

module.exports = {
  SetDrivePower: SetDrivePower,
  GetRobotInfo: GetRobotInfo,
  SetRemoteLoggerLevel: SetRemoteLoggerLevel,
  StartMotion: StartMotion,
  CmdJointTrajectory: CmdJointTrajectory,
  StopMotion: StopMotion,
};
