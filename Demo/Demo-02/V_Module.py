import math

def calculate_angle(a, b, c):
    a = [a.x, a.y]; b = [b.x, b.y]; c = [c.x, c.y]
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians*180.0/math.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def ccw(A, B, C): 
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def check_intersection(A, B, C, D): 
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)