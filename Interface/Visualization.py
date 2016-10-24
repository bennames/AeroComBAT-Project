import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

vertices = (
            (1,-1,-1),
            (1,1,-1),
            (-1,1,-1),
            (-1,-1,-1),
            (1,-1,1),
            (1,1,1),
            (-1,-1,1),
            (-1,1,1),
            )

edges = (
         (0,1),
         (0,3),
         (0,4),
         (2,1),
         (2,3),
         (2,7),
         (6,3),
         (6,4),
         (6,7),
         (5,1),
         (5,4),
         (5,7),
         )

surfaces = (
          (0,1,2,3),
          (3,2,7,6),
          (6,7,5,4),
          (4,5,1,0),
          (1,5,7,2),
          (4,0,3,6),
          )

vertices2 = (
            (-1,-1,-2),
            (0,-1,-2),
            (1,-1,-2),
            (-1,0,-2),
            (0,0,-2),
            (1,0,-2),
            (-1,1,-2),
            (0,1,-2),
            (1,1,-2),
            )
edges2 = (
         (0,1),
         (1,4),
         (4,3),
         (3,0),
         (1,2),
         (2,5),
         (5,4),
         (4,1),
         (4,5),
         (5,8),
         (8,7),
         (7,4),
         (3,4),
         (4,7),
         (7,6),
         (6,3),
         )
surfaces2 = (
          (0,1,4,3),
          (1,2,5,4),
          (4,5,8,7),
          (3,4,7,6),
          )

colors = (
          (1,0,0),
          (0,1,0),
          (0,0,1),
          (1,1,0),
          (0,1,1),
          (1,0,1),
          (1,0,0),
          (0,1,0),
          (0,0,1),
          (1,1,0),
          (0,1,1),
          (1,0,1),
          )
def Cube():
    
    glBegin(GL_QUADS)
    for surface in surfaces:
        x=0
        
        for vertex in surface:
            x+=1
            glColor3fv(colors[x])
            glVertex3fv(vertices[vertex])
    glEnd()
    
    glBegin(GL_LINES)
    for edge in edges:
        glColor3fv((0,0,0))
        for vertex in surface:
            glVertex3fv(vertices[vertex])
    glEnd()

def Mesh():
    glBegin(GL_QUADS)
    for surface in surfaces2:
        x=0
        
        for vertex in surface:
            x+=1
            glColor3fv(colors[x])
            glVertex3fv(vertices2[vertex])
    glEnd()
    
    glBegin(GL_LINES)
    for edge in edges2:
        glColor3fv((0,0,0))
        for vertex in surface:
            glVertex3fv(vertices2[vertex])
    glEnd()
    
def CSYS():
    ORG = (0,0,0)
    XP = (1,0,0)
    YP = (0,1,0)
    ZP = (0,0,1)
    
    glLineWidth(2.0)
    
    glBegin(GL_LINES)
    glColor3f(1,0,0)
    glVertex3fv(ORG)
    glVertex3fv (XP ) 
    glColor3f (0,1,0)
    glVertex3fv (ORG)
    glVertex3fv (YP )
    glColor3f (0,0,1)
    glVertex3fv (ORG)
    glVertex3fv (ZP )
    glEnd()
    
class Model(object):
    distance = 0
    left_key = False
    right_key = False
    up_key = False
    down_key = False
    a_key = False
    s_key = False
    d_key = False
    r_key = False
    f_key = False
    q_key = False
    w_key = False
    e_key = False
    x_axis = 0
    y_axis = 0
    def __init__(self):
        self.coordinates = [0,0,0]

    def render_scene(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glTranslatef(0,0,-5)   
        
        glTranslatef(self.x_axis,self.y_axis,self.distance)
        
        glRotatef(self.coordinates[0],1,0,0)
        glRotatef(self.coordinates[1],0,1,0)
        glRotatef(self.coordinates[2],0,0,1)
        
        CSYS()
        #Cube()
        Mesh()


    def rotate_x(self):
        self.coordinates[0] += 2

    def n_rotate_x(self):
        self.coordinates[0] -= 2
            
    def rotate_y(self):
        self.coordinates[1] += 2

    def n_rotate_y(self):
        self.coordinates[1] -= 2
            
    def rotate_z(self):
        self.coordinates[2] += 2

    def n_rotate_z(self):
        self.coordinates[2] -= 2
            
    def move_away(self):
        self.distance -= 0.1
        
    def move_close(self):
        if self.distance < 0:
            self.distance += 0.1
            
    def move_left(self):
        self.x_axis -= 0.1
        
    def move_right(self):
        self.x_axis += 0.1
        
    def move_up(self):
        self.y_axis += 0.1
        
    def move_down(self):
        self.y_axis -= 0.1
            
    def keydown(self):
        if self.a_key:
            self.rotate_x()
        elif self.s_key:
            self.rotate_y()
        elif self.d_key:
            self.rotate_z()
        elif self.q_key:
            self.n_rotate_x()
        elif self.w_key:
            self.n_rotate_y()
        elif self.e_key:
            self.n_rotate_z()
        elif self.r_key:
            self.move_away()
        elif self.f_key:
            self.move_close()
        elif self.left_key:
            self.move_left()
        elif self.right_key:
            self.move_right()
        elif self.up_key:
            self.move_up()
        elif self.down_key:
            self.move_down()
            
    def keyup(self):
        self.left_key = False
        self.right_key = False
        self.up_key = False
        self.down_key = False
        self.a_key = False
        self.s_key = False
        self.d_key = False
        self.q_key = False
        self.w_key = False
        self.e_key = False
        self.r_key = False
        self.f_key = False
        
def display():
    pygame.init()
    pygame.display.set_mode((640,480),pygame.DOUBLEBUF|pygame.OPENGL)
    pygame.display.set_caption("PyOpenGL Tutorial")
    clock = pygame.time.Clock()
    running = True
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    gluPerspective(45,640.0/480.0,0.1,200.0)
    
    glEnable(GL_DEPTH_TEST)

    model = Model()
    #----------- Main Program Loop -------------------------------------
    while running:
        # --- Main event loop
        for event in pygame.event.get(): # User did something
            if event.type==pygame.QUIT:
                running=False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    model.rotate_x()
                    model.a_key = True
                elif event.key == pygame.K_s:
                    model.rotate_y()
                    model.s_key = True
                elif event.key == pygame.K_d:
                    model.rotate_z()
                    model.d_key = True
                elif event.key ==pygame.K_q:
                    model.n_rotate_x()
                    model.q_key = True
                elif event.key ==pygame.K_w:
                    model.n_rotate_y()
                    model.w_key = True
                elif event.key ==pygame.K_e:
                    model.n_rotate_z()
                    model.e_key = True
                elif event.key == pygame.K_r:
                    model.move_away()
                    model.r_key = True
                elif event.key == pygame.K_f:
                    model.move_close()
                    model.f_key = True
                elif event.key == pygame.K_LEFT:
                    model.move_left()
                    model.left_key = True
                elif event.key == pygame.K_RIGHT:
                    model.move_right()
                    model.right_key = True
                elif event.key == pygame.K_UP:
                    model.move_up()
                    model.up_key = True
                elif event.key == pygame.K_DOWN:
                    model.move_down()
                    model.down_key = True
                    
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    model.keyup()
                elif event.key == pygame.K_s:
                    model.keyup()
                elif event.key == pygame.K_d:
                    model.keyup()
                elif event.key == pygame.K_q:
                    model.keyup()
                elif event.key == pygame.K_w:
                    model.keyup()
                elif event.key == pygame.K_e:
                    model.keyup()
                elif event.key == pygame.K_r:
                    model.keyup()
                elif event.key == pygame.K_f:
                    model.keyup()
                elif event.key == pygame.K_LEFT:
                    model.keyup()
                elif event.key == pygame.K_RIGHT:
                    model.keyup()
                elif event.key == pygame.K_UP:
                    model.keyup()
                elif event.key == pygame.K_DOWN:
                    model.keyup()
        
        model.keydown()
        model.render_scene()
        
        pygame.display.flip()
        clock.tick(30)
    
    #cube.delete_texture()
    pygame.quit()