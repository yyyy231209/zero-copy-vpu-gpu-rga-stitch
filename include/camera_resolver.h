#pragma once

#include <string>

bool resolve_usb_camera_by_hub_port(int downstream_port,
                                    int required_width,
                                    int required_height,
                                    std::string *device_path,
                                    std::string *diagnostic);
