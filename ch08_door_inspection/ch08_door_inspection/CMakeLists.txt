cmake_minimum_required(VERSION 3.8)
project(ch08_door_inspection)

# KARMA PAKET (ament_cmake + ament_cmake_python). Planlama/yürütme başlatma dosyaları ve
# Python modülü ament_python paketindeki gibi kurulur; OctoMap aşaması ise gerçek octomap
# kütüphanesine bağlanması gereken bir C++ programıdır (build_doors_octomap). Kapı işinin
# TEK pakette kalması için ayrı bir C++ paketi açılmamıştır.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)          # std::filesystem ve std::make_unique için
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(octomap REQUIRED)
find_package(Eigen3 REQUIRED)

# ── Python: modül + komut satırı giriş noktaları ────────────────────────────
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/doors_mesh_prep
  scripts/doors_parts_prep
  scripts/doors_mission_node
  scripts/cleaning_mission_runner
  DESTINATION lib/${PROJECT_NAME}
)

# ── C++: OctoMap oluşturucu ─────────────────────────────────────────────────
add_executable(build_doors_octomap src/build_doors_octomap.cpp)
target_include_directories(build_doors_octomap PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)
target_link_libraries(build_doors_octomap
  ${PCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
  Eigen3::Eigen
)

install(TARGETS build_doors_octomap
  DESTINATION lib/${PROJECT_NAME}
)

# ── Veri: başlatma dosyaları, parametreler, kitapta kullanılan hazır plan ───
install(DIRECTORY launch config plans
  DESTINATION share/${PROJECT_NAME}
)

# ── Testler: Python kodu için PEP 8 (flake8) ve PEP 257 (docstring) denetimi ─
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_flake8 test/test_flake8.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  ament_add_pytest_test(test_pep257 test/test_pep257.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

ament_package()
