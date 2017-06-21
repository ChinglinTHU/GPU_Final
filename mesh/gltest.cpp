#include <iostream>
// #define GL_LITE_IMPLEMENTATION
// #include "gl_util.h"
// #define GLFW_DLL
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include <GL/gl.h>
#include "myShader.h"

using namespace std;

GLuint Width = 800, Height = 600;

static void error_callback(int error, const char* description)
{
	cerr<<"GLFW: ("<<error<<") " <<description<<endl;
}

int main()
{
	glfwInit();
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

    glfwSetErrorCallback(error_callback);

    GLFWwindow* window = glfwCreateWindow(Width, Height, "", nullptr, nullptr);
    if(window == nullptr)
    {
    	cout<<"failed to create glfw window"<<endl;
    	glfwTerminate();
    	return -1;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;

    GLenum err = glewInit();
    if(err != GLEW_OK)
    {
    	cout<<"failed to init glew, error code: "<<glewGetErrorString(err)<<endl;
    	return -1;
    }

    glViewport(0,0,Width,Height);

    // loop
    while(!glfwWindowShouldClose(window))
    {
    	glfwPollEvents();

    	glClearColor(0.f,0.f,0.f,1.0f);
    	glClear(GL_COLOR_BUFFER_BIT);

    	glfwSwapBuffers(window);
    }

    glfwTerminate();
	return 0;
}