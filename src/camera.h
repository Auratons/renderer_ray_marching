#ifndef POINTCLOUD_RENDERER_CAMERA_H
#define POINTCLOUD_RENDERER_CAMERA_H

#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "common.h"

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

// Default camera values
const float YAW                   = 0;
const float PITCH                 = 0.0f;
const float SPEED                 = 2.5f;
constexpr float DEG_TO_RAD_FACTOR = PI / 180.0f;
constexpr float SENSITIVITY       = 0.1f * DEG_TO_RAD_FACTOR;
const float ZOOM                  = PI / 4;


// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera
{
public:
    // camera Attributes
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    // euler Angles
    float Yaw;
    float Pitch;
    // camera options
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    // constructor with vectors
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH) :
    Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
    {
      Position = position;
      WorldUp = up;
      Yaw = yaw;
      Pitch = pitch;
      updateCameraVectors();
    }
    // constructor with scalar values
    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
    {
      Position = glm::vec3(posX, posY, posZ);
      WorldUp = glm::vec3(upX, upY, upZ);
      Yaw = yaw;
      Pitch = pitch;
      updateCameraVectors();
    }

    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 GetViewMatrix()
    {
      return glm::lookAt(Position, Position + Front, Up);
    }

    // processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboard(Camera_Movement direction, float deltaTime)
    {
      auto velocity = [] (const glm::vec3 &dir, float speed) { return speed * dir; };
      if (direction == FORWARD)
        Position += velocity(Front, MovementSpeed) * deltaTime;
      if (direction == BACKWARD)
        Position -= velocity(Front, MovementSpeed) * deltaTime;
      if (direction == LEFT)
        Position -= velocity(Right, MovementSpeed) * deltaTime;
      if (direction == RIGHT)
        Position += velocity(Right, MovementSpeed) * deltaTime;
      if (direction == DOWN)
        Position -= velocity(Up, MovementSpeed) * deltaTime;
      if (direction == UP)
        Position += velocity(Up, MovementSpeed) * deltaTime;
    }

    // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
    {
      Yaw   += (xoffset * MouseSensitivity);
      Pitch += (yoffset * MouseSensitivity);

      if (constrainPitch)
      {
        const float abs_max_pitch = PI / 2 - 0.1f;
        if (Pitch > abs_max_pitch)
          Pitch = abs_max_pitch;
        if (Pitch < -abs_max_pitch)
          Pitch = -abs_max_pitch;
      }

      // update Front, Right and Up Vectors using the updated Euler angles
      updateCameraVectors();
    }

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void ProcessMouseScroll(float yoffset)
    {
      Zoom -= (yoffset * DEG_TO_RAD_FACTOR);
      if (Zoom < 0.1f)
        Zoom = 0.1f;
      if (Zoom > ZOOM)
        Zoom = ZOOM;
    }

private:
    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors()
    {
      auto default_look = glm::vec4(0, 0, -1, 1);
      auto pitch_rot = glm::rotate(glm::mat4(1.0f), Pitch, glm::vec3(1, 0, 0));
      auto yaw_rot = glm::rotate(glm::mat4(1.0f), Yaw, glm::vec3(0, 1, 0));
      Front = glm::normalize(glm::vec3(yaw_rot * pitch_rot * default_look));
      Right = glm::normalize(glm::cross(Front, WorldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
      Up    = glm::normalize(glm::cross(Right, Front));
    }
};

#endif //POINTCLOUD_RENDERER_CAMERA_H
