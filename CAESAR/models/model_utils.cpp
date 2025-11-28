#include "model_utils.h"
#include <cstdlib>
#include <stdexcept>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

fs::path get_executable_path() {
#ifdef _WIN32
    char result[MAX_PATH];
    DWORD count = GetModuleFileNameA(NULL , result , MAX_PATH);
    if (count != 0 && count < MAX_PATH) {
        return fs::path(result);
    }
#elif defined(__linux__)
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe" , result , PATH_MAX);
    if (count != -1) {
        return fs::path(std::string(result , count));
    }
#elif defined(__APPLE__)
    char result[PATH_MAX];
    uint32_t size = sizeof(result);
    if (_NSGetExecutablePath(result , &size) == 0) {
        return fs::canonical(fs::path(result));
    }
#endif
    throw std::runtime_error("Unable to determine executable path");
}

fs::path get_model_file(const std::string& filename) {
    const char* env_p = std::getenv("CAESAR_MODEL_DIR");
    if (env_p) {
        fs::path model_path = fs::path(env_p) / filename;
        if (fs::exists(model_path)) {
            return model_path;
        }
        std::cerr << "Warning: CAESAR_MODEL_DIR is set but file not found at: "
            << model_path << std::endl;
    }

    try {
        fs::path exe_path = get_executable_path();
        fs::path exe_dir = exe_path.parent_path();

        fs::path model_path = exe_dir / "exported_model" / filename;
        if (fs::exists(model_path)) {
            return fs::canonical(model_path);
        }

        model_path = exe_dir / ".." / "exported_model" / filename;
        if (fs::exists(model_path)) {
            return fs::canonical(model_path);
        }

        model_path = exe_dir / ".." / ".." / "exported_model" / filename;
        if (fs::exists(model_path)) {
            return fs::canonical(model_path);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Warning: Could not check executable-relative path: "
            << e.what() << std::endl;
    }

#ifdef DEFAULT_CAESAR_MODEL_DIR
    fs::path install_path = fs::path(DEFAULT_CAESAR_MODEL_DIR) / filename;
    if (fs::exists(install_path)) {
        return install_path;
    }
#endif

    throw std::runtime_error(
        "Could not find model file: " + filename + "\n"
        "Searched locations:\n"
        "  1. CAESAR_MODEL_DIR environment variable" +
        std::string(env_p ? " (" + std::string(env_p) + ")" : " (not set)") + "\n"
        "  2. ../exported_model/ relative to executable\n"
        "  3. ./exported_model/ relative to executable\n"
#ifdef DEFAULT_CAESAR_MODEL_DIR
        "  4. Install location: " + std::string(DEFAULT_CAESAR_MODEL_DIR) + "\n"
#endif
        "\nPlease set CAESAR_MODEL_DIR to point to your exported_model directory."
    );
}