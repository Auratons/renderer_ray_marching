#include <cstdlib>
#include <chrono>
#include <exception>
#include <iostream>
#include <filesystem>
#include <map>  // Even thought seems unused it's needed for nlohmann
#include <string>

#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"  // Even thought seems unused it's needed
#include "CLI/Config.hpp"  // Even thought seems unused it's needed
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <kdtree/kdtree_flann.h>
#include <nlohmann/json.hpp>
#include <thrust/device_vector.h>

#include "camera.h"
#include "common.h"
#include "egl.hpp"
#include "pointcloud.h"
#include "quad.h"
#include "ray_marching.h"
#include "shader.h"
#include "texture.h"
#include "utils.h"


glm::vec2 last_mouse_pos{SCREEN_WIDTH / 2.0f, SCREEN_HEIGHT / 2.0f};
float deltaTime = 0.0f;
float lastFrame = 0.0f;
bool mouse_hold = false;
auto camera = Camera(glm::vec3(0.0f, 0.0f, 3.0f));

using namespace std;
using json = nlohmann::json;
using namespace std::chrono;

GLFWwindow* init_glfw();
void        framebuffer_size_callback(GLFWwindow *, int width, int height);
void        mouse_callback(GLFWwindow* window, double xpos, double ypos);
void        scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void        process_input(GLFWwindow *window);


int main(int argc, char** argv) {
  string pcd_path, matrix_path, output_path;
  bool headless = false, precompute = false;
  CLI::App args{"Pointcloud Renderer"};
  auto file = args.add_option("-f,--file", pcd_path, "Path to pointcloud to render");
  args.add_option("-m,--matrices", matrix_path, "Path to view matrices json for which to render pointcloud in case of headless rendering.");
  args.add_option("-o,--output_path", output_path, "Path where to store renders in case of headless rendering.");
  args.add_flag("-d,--headless", headless, "Run headlessly without a window");
  args.add_flag("-p,--precompute-radii", precompute, "Precompute radii even if already precomputed.");
  file->required();
  CLI11_PARSE(args, argc, argv);

  auto output = filesystem::path(output_path);
  auto [vertices_host, colors_host] = Pointcloud::load_ply(pcd_path);
//  vertices_host.resize(10000);
//  colors_host.resize(10000);
  auto vertices = thrust::device_vector<glm::vec4>(vertices_host.begin(), vertices_host.end());
  auto colors = thrust::device_vector<glm::vec4>(colors_host.begin(), colors_host.end());

  auto points = thrust::device_vector<float>(thrust::device_ptr<float>((float*)vertices.data().get()), thrust::device_ptr<float>((float*)vertices.data().get()) + 4 * vertices.size());
  auto query = thrust::device_vector<float>(points.begin(), points.end());

  cout << "Number of points: " << vertices.size() << endl;
//  cout << "Min radius: " << *std::min_element(radii.begin(), radii.end()) << endl;
//  cout << "Max radius: " << *std::max_element(radii.begin(), radii.end()) << endl;
//  cout << "Avg radius: " << std::accumulate(radii.begin(), radii.end(), 0.0f) / (float)radii.size() << endl;

  thrust::device_vector<int> indices;
  thrust::device_vector<float> radii;
  vector<float> radii_host;
  kdtree::KDTreeSearchParams search_params(2);
  auto tree = kdtree::KDTreeFlann();
  tree.Build(points);

  auto radii_path = filesystem::path(pcd_path + ".kdtree.radii");
  if (!exists(radii_path) || precompute) {
    cout << "Precomputing radii..." << endl;
    tree.Search(query, search_params, indices, radii);
    for (size_t i = 1; i < indices.size(); i += 2) {
      radii[i / 2] = std::sqrt(radii[i]);  // KD tree returns squared distance
    }
    radii.resize(radii.size() / 2);

    radii_host.resize(radii.size());
    thrust::copy(radii.begin(), radii.end(), radii_host.begin());
    std::ofstream ofs(pcd_path + ".kdtree.radii");
    boost::archive::text_oarchive oa(ofs);
    oa & radii_host;
    ofs.close();
  }
  else {
    cout << "Using precomputed radii..." << endl;
    std::ifstream ifs(radii_path);
    boost::archive::text_iarchive ia(ifs);
    ia & radii_host;
    ifs.close();
    radii.resize(radii_host.size());
    thrust::copy(radii_host.begin(), radii_host.end(), radii.begin());
    if (radii.size() != vertices.size()) {
      throw runtime_error(string("The radii file '") + radii_path.string() + string(
              "' does not contain the same number of points as PLY file '" + pcd_path + "'.")
      );
    }
  }

  GLFWwindow* window;
  EGLDisplay display;

  auto successful_run = true;
  try {
    if (headless) {
      display = init_egl();
      init_glad((GLADloadproc)eglGetProcAddress);

      if (!matrix_path.empty()) {
        std::ifstream matrices{matrix_path};
        if (matrices.good()) {
          cout << "Matrices loaded." << endl;
          json j;
          matrices >> j;
          auto process = [&](const string &target_render_path, const json &params) {
            auto path = filesystem::path(target_render_path);
            auto last_but_one_segment = *(--(--path.end()));
            auto last_segment = *(--path.end());
            auto output_file_path = output / last_but_one_segment / last_segment;
            auto lock_file_path = output / last_but_one_segment / ("." + last_segment.string() + ".lock");
            if (!exists(output)) std::filesystem::create_directory(output);
            if (!exists(output / last_but_one_segment)) filesystem::create_directory(output / last_but_one_segment);
            if (filesystem::exists(output_file_path)) {
              cout << canonical(absolute(output_file_path)) << ": " << "ALREADY EXISTS" << endl;
              return;
            }
            { ofstream{lock_file_path}; }
            boost::interprocess::file_lock lock(lock_file_path.c_str());
            if (!lock.try_lock()) {
              cout << absolute(output_file_path) << ": " << "ALREADY LOCKED" << endl;
              return;
            }
            cout << absolute(output_file_path) << ": " << "LOCKING" << endl;
            auto camera_pose = params.at("extrinsic_matrix").get<glm::mat4>();
            auto camera_matrix = params.at("intrinsic_matrix").get<glm::mat4>();
            auto image_width = 2.0f * camera_matrix[2][0];
            auto image_height = 2.0f * camera_matrix[2][1];
            auto focal_length_pixels = camera_matrix[0][0];
            assert(focal_length_pixels == camera_matrix[1][1]);
            auto fov_radians = 2.0f * atanf(image_width / (2.0f * focal_length_pixels));

            auto texture = Texture2D((GLsizei)image_width, (GLsizei)image_height, nullptr, GL_RGBA32F, GL_RGBA, GL_CLAMP_TO_EDGE, GL_NEAREST);
            auto ray_marcher = PointcloudRayMarcher::get_instance(vertices, colors, radii, texture, tree);

            auto start = high_resolution_clock::now();
            ray_marcher->render_to_texture(camera_pose, fov_radians);
            auto end = high_resolution_clock::now();
            ray_marcher->save_png(output_file_path.c_str());
            cout << canonical(absolute(output_file_path)) << ": " << (float)duration_cast<milliseconds>(end - start).count() / 1000.0f << " s" << endl;

            lock.unlock();
            remove(lock_file_path);
          };
          for (auto &[target_render_path, params]: j.at("train").items()) {
            process(target_render_path, params);
          }
          for (auto &[target_render_path, params]: j.at("val").items()) {
            process(target_render_path, params);
          }
        }
      }
    }
    else {
      window = init_glfw();
      init_glad((GLADloadproc)glfwGetProcAddress);

      glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      auto shader_path_vertex = filesystem::current_path() / "shaders/vertex.glsl";
      auto shader_path_fragment = filesystem::current_path() / "shaders/fragment.glsl";
      auto shader_graphical = Shader({shader_path_vertex, shader_path_fragment}, {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER});
      auto quad = Quad(shader_graphical);

      FPSCounter fps;
      camera.Zoom = glm::radians(60.0f);
      auto texture = Texture2D(SCREEN_WIDTH, SCREEN_HEIGHT, nullptr, GL_RGBA32F, GL_RGBA, GL_CLAMP_TO_EDGE, GL_NEAREST);
      auto ray_marcher = PointcloudRayMarcher::get_instance(vertices, colors, radii, texture, tree);

      // render loop
      while (!glfwWindowShouldClose(window)) {
        auto currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        ray_marcher->render_to_texture(camera.GetViewMatrix(), camera.Zoom);
        quad.render(ray_marcher->get_texture().get_id());

        fps.update();
        process_input(window);

        glfwSwapBuffers(window);
        glfwPollEvents();
      }
    }
  }
  catch (const std::exception &e) {
    cerr << e.what() << endl;
    successful_run = false;
  }
  if (!headless) glfwTerminate(); else eglTerminate(display);
  return (successful_run) ? EXIT_SUCCESS : EXIT_FAILURE;
}

GLFWwindow* init_glfw() {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  auto window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Pointcloud Renderer", nullptr, nullptr);
  if (!window) {
    std::cerr << "Fatal error: Failed to create GLFW window." << std::endl;
    exit(1);
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  // Ensure we can capture the escape key being pressed
  glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
  return window;
}

void framebuffer_size_callback(GLFWwindow *, int width, int height) {
  glViewport(0, 0, width, height);
}

void process_input(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.ProcessKeyboard(FORWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.ProcessKeyboard(BACKWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.ProcessKeyboard(LEFT, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.ProcessKeyboard(RIGHT, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    camera.ProcessKeyboard(DOWN, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    camera.ProcessKeyboard(UP, deltaTime);
}

void mouse_callback(GLFWwindow *window, double xposIn, double yposIn) {
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
    glm::vec2 current_mouse_pos;
    current_mouse_pos = glm::vec2(static_cast<float>(xposIn), static_cast<float>(yposIn));
    if (!mouse_hold) {
      last_mouse_pos = current_mouse_pos;
      mouse_hold = true;
    }

    auto offset = current_mouse_pos - last_mouse_pos;
    last_mouse_pos = current_mouse_pos;

    camera.ProcessMouseMovement(offset.x, offset.y);
  } else {
    mouse_hold = false;
  }
}

void scroll_callback(GLFWwindow*, double, double yoffset) {
  camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
