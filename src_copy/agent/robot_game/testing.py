import serial
import time

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



    while True:
        try:
            user_input = input("Enter servo command (e.g., 1010 for Servo 1 -> 10°): ")
            
            # if I can control with arrow keys

            # use arrow keys to determine the next position


            # what command will it translated to 


            
            
            
            if user_input.lower() == "exit":
                break
            if user_input.strip() == "0":
                # Request servo positions from Arduino
                ser.write(b"0\n")  # Tell Arduino to send positions
                print("Requested servo positions...")
                for i in range(3):
                    response = ser.readline().decode().strip()
                    print(f"Arduino: {response}")
                time.sleep(1)
            if user_input.isdigit():
                ser.write((user_input + '\n').encode())
                print(f"Sent: {user_input}")
            else:
                print("Please enter a valid number (or 'exit' to quit).")
        except KeyboardInterrupt:
            break

    ser.close()
