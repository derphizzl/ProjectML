set(LOGGING_FEATURE_ENABLED ON)

list(APPEND DEFINES LOGGING_FEATURE_ENABLED=1)
list(APPEND HEADERS ${CMAKE_CURRENT_LIST_DIR}/Logging.h)
list(APPEND INCPATHS ${CMAKE_CURRENT_LIST_DIR})
