import cv2
import numpy as np
import random
import os
import argparse

def sample_branches(branch_prob):
    branches = 1
    while random.random() < branch_prob:
        branches += 1
    return branches

def draw_variable_crack(mask, segment_mask, start_point, branches=3, length=100,
                        thickness_scale=1.0, initial_angle=None, max_retries=5):
    """
    Draw cracks starting from start_point. If a step goes outside segmented area,
    it tries other directions up to max_retries.
    """
    end_points = []
    for _ in range(branches):
        x, y = start_point
        angle = initial_angle if initial_angle is not None else random.uniform(0, 2 * np.pi)

        for i in range(length):
            for retry in range(max_retries):
                dx = int(np.cos(angle) * random.randint(1, 3))
                dy = int(np.sin(angle) * random.randint(1, 3))
                nx = np.clip(x + dx + random.randint(-1, 1), 0, mask.shape[1]-1)
                ny = np.clip(y + dy + random.randint(-1, 1), 0, mask.shape[0]-1)

                if segment_mask[ny, nx] > 0:
                    x, y = nx, ny
                    break
                else:
                    angle += random.uniform(-np.pi/4, np.pi/4)  # try a new direction
            else:
                break  # stop if all retries fail

            thickness = int((1 + 2 * np.sin(np.pi * i / length)) * thickness_scale)
            thickness = max(1, thickness)
            cv2.circle(mask, (x, y), thickness, 255, -1)

            angle += random.uniform(-0.2, 0.2)
        end_points.append((x, y, angle))

    # Extend each branch slightly
    for (ex, ey, angle) in end_points:
        for _ in range(40):
            for retry in range(max_retries):
                dx = int(np.cos(angle) * 2)
                dy = int(np.sin(angle) * 2)
                nx = np.clip(ex + dx, 0, mask.shape[1]-1)
                ny = np.clip(ey + dy, 0, mask.shape[0]-1)

                if segment_mask[ny, nx] > 0:
                    ex, ey = nx, ny
                    break
                else:
                    angle += random.uniform(-np.pi/6, np.pi/6)
            else:
                break

            cv2.circle(mask, (ex, ey), max(1, int(thickness_scale)), 255, -1)
            angle += random.uniform(-0.05, 0.05)

    return mask

def generate_crack_mask(segment_mask, num_cracks, min_length, max_length, branch_prob, thickness_scale):
    mask = np.zeros_like(segment_mask, dtype=np.uint8)
    height, width = segment_mask.shape

    # Detect edges of segmented area
    edges = cv2.Canny(segment_mask, 100, 200)
    edge_points = list(zip(*np.where(edges > 0)))

    # If no edges (fully segmented), use image border
    if len(edge_points) == 0:
        print("No edges detected. Using image border as start points.")
        edge_points = []
        for x in range(width):
            edge_points.append((x, 0))
            edge_points.append((x, height-1))
        for y in range(height):
            edge_points.append((0, y))
            edge_points.append((width-1, y))

    for _ in range(num_cracks):
        # Ensure start is inside segmentation
        for attempt in range(20):
            start_point = random.choice(edge_points)
            if segment_mask[start_point[1], start_point[0]] > 0:
                break
        else:
            continue

        length = random.randint(min_length, max_length)
        branches = sample_branches(branch_prob)
        x, y = start_point

        # Bias initial angle inward if starting on image border
        if x == 0:
            angle = random.uniform(-np.pi/4, np.pi/4)
        elif x == width-1:
            angle = random.uniform(3*np.pi/4, 5*np.pi/4)
        elif y == 0:
            angle = random.uniform(np.pi/4, 3*np.pi/4)
        elif y == height-1:
            angle = random.uniform(-3*np.pi/4, -np.pi/4)
        else:
            angle = random.uniform(0, 2 * np.pi)

        mask = draw_variable_crack(mask, segment_mask, start_point,
                                   branches=branches, length=length,
                                   thickness_scale=thickness_scale,
                                   initial_angle=angle)

    return mask

def main():
    parser = argparse.ArgumentParser(description="Generate cracks on a binary segment mask.")
    parser.add_argument("input", type=str)
    parser.add_argument("--num_cracks", type=int, default=3)
    parser.add_argument("--min_length", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--branch_prob", type=float, default=0.3)
    parser.add_argument("--thickness_scale", type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("File does not exist.")
        return

    segment_mask = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if segment_mask is None:
        print("Failed to read image. Make sure it is a valid image file.")
        return

    _, segment_mask = cv2.threshold(segment_mask, 127, 255, cv2.THRESH_BINARY)

    crack_mask = generate_crack_mask(segment_mask,
                                     num_cracks=args.num_cracks,
                                     min_length=args.min_length,
                                     max_length=args.max_length,
                                     branch_prob=args.branch_prob,
                                     thickness_scale=args.thickness_scale)

    base_name = os.path.splitext(args.input)[0]
    output_path = base_name.replace("cropped_mask", "") + "_crack_mask.png"

    cv2.imwrite(output_path, crack_mask)
    print(f"Crack mask saved to {output_path}")

if __name__ == "__main__":
    main()
