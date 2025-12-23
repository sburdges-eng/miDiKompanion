#pragma once

/**
 * Platform compatibility layer for Penta-Core
 * Handles macOS SDK compatibility issues and cross-platform definitions
 */

// macOS SDK 26.2+ compatibility fix
// Must be included before any system headers to avoid __CLOCK_AVAILABILITY issues
#if (defined(__APPLE__) || defined(__APPLE_CPP__) || defined(__APPLE_CC__))
 #ifndef TARGET_OS_MAC
  #include <TargetConditionals.h>
 #endif

 // Define SDK compatibility macros before system headers
 #ifndef __CLOCK_AVAILABILITY
  #define __CLOCK_AVAILABILITY
 #endif
 #ifndef __API_AVAILABLE
  #define __API_AVAILABLE(...)
 #endif
 #ifndef __API_DEPRECATED
  #define __API_DEPRECATED(...)
 #endif
 #ifndef __API_UNAVAILABLE
  #define __API_UNAVAILABLE(...)
 #endif
 #ifndef __OSX_AVAILABLE
  #define __OSX_AVAILABLE(...)
 #endif
 #ifndef __OSX_AVAILABLE_STARTING
  #define __OSX_AVAILABLE_STARTING(...)
 #endif
 #ifndef __WATCHOS_PROHIBITED
  #define __WATCHOS_PROHIBITED
 #endif
 #ifndef __TVOS_PROHIBITED
  #define __TVOS_PROHIBITED
 #endif
 #ifndef __IOS_AVAILABLE
  #define __IOS_AVAILABLE(...)
 #endif
 #ifndef __TVOS_AVAILABLE
  #define __TVOS_AVAILABLE(...)
 #endif
 #ifndef __WATCHOS_AVAILABLE
  #define __WATCHOS_AVAILABLE(...)
 #endif
#endif

// Standard includes that are safe after the above definitions
#include <cstdint>
#include <cstddef>
