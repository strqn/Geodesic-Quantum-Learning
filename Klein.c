#include <GL/glut.h>
#include <cmath>
#include <vector>
#include <cstdlib> // for rand()

// Window dimensions
const int WIDTH = 800;
const int HEIGHT = 600;

// Grid dimensions for the 3D surface
const int GRID_SIZE = 100;          // number of points per dimension
const float L = 20.0f;              // spatial extent from -L to L
const float GRID_STEP = 2.0f * L / (GRID_SIZE - 1);

// Time evolution
float t = 0.0f;                     // current time
float timeSpeed = 0.05f;             // speed of animation
bool isAnimating = true;             // pause/unpause

// Camera rotation
float rotX = 30.0f, rotY = 0.0f;    // rotation angles
int lastMouseX = -1, lastMouseY = -1;
bool mouseRotating = false;

// Plane wave structure for the Klein-Gordon equation
struct PlaneWave {
    float kx, ky;   // wave vector components (momentum)
    float m;        // mass
    float A;        // amplitude

    // Calculates the angular frequency (energy) from the mass shell condition
    float omega() const {
        return std::sqrt(kx*kx + ky*ky + m*m);
    }

    // Evaluates the real part of the wave function at (x, y, t)
    float evaluate(float x, float y, float t) const {
        float phase = kx * x + ky * y - omega() * t;
        return A * std::cos(phase);
    }
};

std::vector<PlaneWave> waves;

// Grid data: stores (x, y, z) for each point
struct GridPoint {
    float x, y, z;
};
std::vector<std::vector<GridPoint>> grid(GRID_SIZE, std::vector<GridPoint>(GRID_SIZE));

// Function to recompute the grid values at current time t
void updateGrid() {
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            float x = -L + i * GRID_STEP;
            float y = -L + j * GRID_STEP;
            float z = 0.0f;

            // Superimpose all waves
            for (const auto& wave : waves) {
                z += wave.evaluate(x, y, t);
            }

            grid[i][j] = {x, y, z};
        }
    }
}

// Initialize OpenGL settings and the wave packet superposition
void init() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL); // so that glColor affects material properties
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    // Set up a simple directional light
    GLfloat lightPos[] = {1.0f, 1.0f, 1.0f, 0.0f};
    GLfloat lightAmbient[] = {0.2f, 0.2f, 0.2f, 1.0f};
    GLfloat lightDiffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);

    // Create a set of plane waves with random parameters for richer pattern
    waves.clear();
    // parameters: {kx, ky, mass, amplitude}
    waves.push_back({1.0f, 0.5f, 1.5f, 1.0f});
    waves.push_back({-0.5f, 1.2f, 1.5f, 0.8f});
    waves.push_back({0.2f, -1.0f, 1.5f, 0.6f});
    waves.push_back({0.8f, -0.7f, 1.2f, 0.7f});
    waves.push_back({-0.9f, -0.3f, 1.8f, 0.5f});
    // Add a couple more for complexity
    for (int i = 0; i < 3; ++i) {
        float kx = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        float ky = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        float m  = 1.0f + ((float)rand() / RAND_MAX) * 1.0f;
        float A  = 0.3f + ((float)rand() / RAND_MAX) * 0.5f;
        waves.push_back({kx, ky, m, A});
    }

    // Initialize grid with t=0
    updateGrid();
}

// Draw the surface as a solid mesh with smooth shading and wireframe overlay
void drawWaveSurface() {
    glPushMatrix();

    // Apply camera transformations (rotate based on mouse)
    glTranslatef(0.0f, -5.0f, -40.0f);
    glRotatef(rotX, 1.0f, 0.0f, 0.0f);
    glRotatef(rotY, 0.0f, 1.0f, 0.0f);

    // Draw filled triangles with lighting
    glEnable(GL_LIGHTING);
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < GRID_SIZE - 1; ++i) {
        for (int j = 0; j < GRID_SIZE - 1; ++j) {
            // Get the four corners of the grid cell
            const GridPoint& p00 = grid[i][j];
            const GridPoint& p10 = grid[i+1][j];
            const GridPoint& p01 = grid[i][j+1];
            const GridPoint& p11 = grid[i+1][j+1];

            // Compute normal for each triangle (approximate by cross product)
            // First triangle (p00, p10, p11)
            float nx1 = (p10.y - p00.y)*(p11.z - p00.z) - (p10.z - p00.z)*(p11.y - p00.y);
            float ny1 = (p10.z - p00.z)*(p11.x - p00.x) - (p10.x - p00.x)*(p11.z - p00.z);
            float nz1 = (p10.x - p00.x)*(p11.y - p00.y) - (p10.y - p00.y)*(p11.x - p00.x);
            // Normalize
            float len1 = std::sqrt(nx1*nx1 + ny1*ny1 + nz1*nz1);
            if (len1 > 0) { nx1 /= len1; ny1 /= len1; nz1 /= len1; }

            // Second triangle (p00, p11, p01)
            float nx2 = (p11.y - p00.y)*(p01.z - p00.z) - (p11.z - p00.z)*(p01.y - p00.y);
            float ny2 = (p11.z - p00.z)*(p01.x - p00.x) - (p11.x - p00.x)*(p01.z - p00.z);
            float nz2 = (p11.x - p00.x)*(p01.y - p00.y) - (p11.y - p00.y)*(p01.x - p00.x);
            float len2 = std::sqrt(nx2*nx2 + ny2*ny2 + nz2*nz2);
            if (len2 > 0) { nx2 /= len2; ny2 /= len2; nz2 /= len2; }

            // Color based on height (z) - blue for low, red for high
            float z00 = p00.z, z10 = p10.z, z01 = p01.z, z11 = p11.z;
            float colorFactor00 = (z00 + 3.0f) / 6.0f; // map typical range -3..3 to 0..1
            float colorFactor10 = (z10 + 3.0f) / 6.0f;
            float colorFactor01 = (z01 + 3.0f) / 6.0f;
            float colorFactor11 = (z11 + 3.0f) / 6.0f;
            // Clamp
            colorFactor00 = (colorFactor00 < 0) ? 0 : (colorFactor00 > 1) ? 1 : colorFactor00;
            colorFactor10 = (colorFactor10 < 0) ? 0 : (colorFactor10 > 1) ? 1 : colorFactor10;
            colorFactor01 = (colorFactor01 < 0) ? 0 : (colorFactor01 > 1) ? 1 : colorFactor01;
            colorFactor11 = (colorFactor11 < 0) ? 0 : (colorFactor11 > 1) ? 1 : colorFactor11;

            // First triangle
            glNormal3f(nx1, ny1, nz1);
            glColor3f(colorFactor00, 0.2f, 1.0f - colorFactor00);
            glVertex3f(p00.x, p00.y, p00.z);
            glColor3f(colorFactor10, 0.2f, 1.0f - colorFactor10);
            glVertex3f(p10.x, p10.y, p10.z);
            glColor3f(colorFactor11, 0.2f, 1.0f - colorFactor11);
            glVertex3f(p11.x, p11.y, p11.z);

            // Second triangle
            glNormal3f(nx2, ny2, nz2);
            glColor3f(colorFactor00, 0.2f, 1.0f - colorFactor00);
            glVertex3f(p00.x, p00.y, p00.z);
            glColor3f(colorFactor11, 0.2f, 1.0f - colorFactor11);
            glVertex3f(p11.x, p11.y, p11.z);
            glColor3f(colorFactor01, 0.2f, 1.0f - colorFactor01);
            glVertex3f(p01.x, p01.y, p01.z);
        }
    }
    glEnd();

    // Draw wireframe overlay in white (disable lighting for lines)
    glDisable(GL_LIGHTING);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_LINES);
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            const GridPoint& p = grid[i][j];
            if (i < GRID_SIZE - 1) {
                const GridPoint& pNext = grid[i+1][j];
                glVertex3f(p.x, p.y, p.z);
                glVertex3f(pNext.x, pNext.y, pNext.z);
            }
            if (j < GRID_SIZE - 1) {
                const GridPoint& pNext = grid[i][j+1];
                glVertex3f(p.x, p.y, p.z);
                glVertex3f(pNext.x, pNext.y, pNext.z);
            }
        }
    }
    glEnd();

    glPopMatrix();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    drawWaveSurface();

    glutSwapBuffers();
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)w / (double)h, 1.0, 200.0);
    glMatrixMode(GL_MODELVIEW);
}

void update(int value) {
    if (isAnimating) {
        t += timeSpeed;
        updateGrid();   // recompute grid with new time
    }
    glutPostRedisplay();
    glutTimerFunc(16, update, 0);   // ~60 FPS
}

// Keyboard interaction
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case ' ':
            isAnimating = !isAnimating;
            break;
        case '+':
        case '=':
            timeSpeed += 0.01f;
            break;
        case '-':
            timeSpeed -= 0.01f;
            if (timeSpeed < 0.0f) timeSpeed = 0.0f;
            break;
        case 'r':
        case 'R':
            rotX = 30.0f; rotY = 0.0f;
            break;
        case 27: // ESC
            exit(0);
            break;
        default:
            break;
    }
}

// Mouse interaction for rotation
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            mouseRotating = true;
            lastMouseX = x;
            lastMouseY = y;
        } else {
            mouseRotating = false;
        }
    }
}

void motion(int x, int y) {
    if (mouseRotating) {
        rotY += (x - lastMouseX) * 0.5f;
        rotX += (y - lastMouseY) * 0.5f;
        lastMouseX = x;
        lastMouseY = y;
        glutPostRedisplay();
    }
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Klein-Gordon 3D Wave Visualization");

    init();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutTimerFunc(16, update, 0);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    glutMainLoop();
    return 0;
}
