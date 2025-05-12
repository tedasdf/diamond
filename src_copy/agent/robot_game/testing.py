import copy
import serial
import time
import pygame
import modern_robotics as mr

L1 = 1.0  # Length of the first arm segment
L2 = 1.0  # Length of the second arm segment
L3 = 1.0  # Length of the third arm segment

M = 


def forward_kinematics(theta):
    Tsb = mr.FKinBody(M, Blist, theta)



def send_angles(theta1, theta2, theta3):
    # Send angles to Arduino
    ser.write(f"{theta1},{theta2},{theta3}\n".encode())
    print(f"Sent angles: {theta1}, {theta2}, {theta3}")




if __name__ == "__main__":
    ser = serial.Serial('COM3', 9600, timeout=1, write_timeout=1)
    time.sleep(2)  # Give time for the connection to establish


    motor_degree = [0 , 0 ,0]
    # before setarting calculating the current position pf the end effector 

    
    ser.write(b"0\n")  # Tell Arduino to send positions
    print("Requested servo positions...")

    for i in range(3):
        response = ser.readline().decode().strip()
        print(f"Arduino: {response}")

    current_pos = forward_kinematics(motor_degree)
    step = 0.1

    while True:
        try:
            keys = pygame.key.get_pressed()
            next_pos = copy.deepcopy(current_pos)
            # if I can control with arrow keys
            if keys[pygame.K_UP]:
                next_pos[0] += step
            elif keys[pygame.K_DOWN]:
                next_pos[0] -= step
            elif keys[pygame.K_LEFT]:
                next_pos[1] -= step
            elif keys[pygame.K_RIGHT]:
                next_pos[1] += step
            elif keys[pygame.K_w]:
                next_pos[2] += step
            elif keys[pygame.K_s]:
                next_pos[2] -= step
            elif keys[pygame.K_0]:
                # Request servo positions from Arduino
                ser.write(b"0\n")  # Tell Arduino to send positions
                print("Requested servo positions...")
                for i in range(3):
                    response = ser.readline().decode().strip()
                    print(f"Arduino: {response}")
            # use arrow keys to determine the next position
            theta1, theta2 , theta3 = inverse_kinematics(current_pos,next_pos)
            send_angles(theta1, theta2, theta3)
            # what command will it translated to 


            
                
            # if user_input.isdigit():
            #     ser.write((user_input + '\n').encode())
            #     print(f"Sent: {user_input}")
            # else:
            #     print("Please enter a valid number (or 'exit' to quit).")
            time.sleep(0.5)
        except KeyboardInterrupt:
            break

    ser.close()
