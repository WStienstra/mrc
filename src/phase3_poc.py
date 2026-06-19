#!/usr/bin/env python3
"""
Wietse Fractal Compression — Phase 3 Proof of Concept

1. Generate the fractal registry (65,536 entries, 256×256 grid)
2. Read a 64-byte file → 32 × 16-bit values
3. For each value, find ALL positions on the fractal grid
4. Search for a geometric pattern connecting 32 positions (one per value)
5. Store: starting position + pattern = less than 64 bytes
"""

import struct
import os
import sys
from collections import defaultdict
import itertools
import math

# ============================================================
# Step 1: Generate the fractal registry (same as Rust version)
# ============================================================

def mandelbrot(cx, cy, max_iter):
    zx, zy = 0.0, 0.0
    for i in range(max_iter):
        if zx*zx + zy*zy > 4.0:
            return i
        zx, zy = zx*zx - zy*zy + cx, 2*zx*zy + cy
    return max_iter

def anti_selfsimilar_hash(x, y, iter_count):
    """Same mixing function as the Rust version."""
    h = iter_count & 0xFFFFFFFFFFFFFFFF
    
    # Round 1: fold in x
    h = (h * 2654435761) & 0xFFFFFFFFFFFFFFFF
    h ^= (x * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    h = ((h << 17) | (h >> 47)) & 0xFFFFFFFFFFFFFFFF
    
    # Round 2: fold in y
    h ^= (y * 0x517CC1B727220A95) & 0xFFFFFFFFFFFFFFFF
    h = (h * 0x6C62272E07BB0142) & 0xFFFFFFFFFFFFFFFF
    h = ((h << 31) | (h >> 33)) & 0xFFFFFFFFFFFFFFFF
    
    # Round 3: avalanche
    h ^= (h >> 33)
    h = (h * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    h = (h * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    
    return h & 0xFFFF

def build_registry(width=256, height=256, max_iter=65536):
    """Build the fractal registry and return grid + reverse lookup."""
    print(f"Building {width}×{height} Mandelbrot registry (max_iter={max_iter})...")
    
    # Main grid
    grid = {}  # (x,y) → value
    value_to_positions = defaultdict(list)  # value → [(x,y), ...]
    
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    
    for py in range(height):
        cy = y_min + py * (y_max - y_min) / height
        for px in range(width):
            cx = x_min + px * (x_max - x_min) / width
            iter_count = mandelbrot(cx, cy, max_iter)
            value = anti_selfsimilar_hash(px, py, iter_count)
            grid[(px, py)] = value
            value_to_positions[value].append((px, py))
    
    covered = set(grid.values())
    print(f"  Base pass: {len(covered)}/65536 unique values")
    
    # Additional zoom passes for coverage
    zoom_targets = [
        (-0.75, 0.1, 0.5), (-1.25, 0.0, 0.3), (-0.16, 1.04, 0.1),
        (-1.75, 0.0, 0.05), (0.28, 0.53, 0.02), (-0.745, 0.186, 0.01),
        (-0.5, 0.56, 0.15), (-1.0, 0.3, 0.2), (-0.4, -0.6, 0.1),
        (-0.108, 0.89, 0.03), (-1.77, 0.0, 0.01), (0.36, 0.1, 0.01),
        (-0.6, 0.0, 0.5), (-1.5, 0.0, 0.5), (-0.2, 0.8, 0.3),
    ]
    
    extra_idx = width * height
    for cx_z, cy_z, span in zoom_targets:
        if len(covered) >= 65536:
            break
        for py in range(height):
            cy = cy_z - span + py * 2*span / height
            for px in range(width):
                cx = cx_z - span + px * 2*span / width
                iter_count = mandelbrot(cx, cy, max_iter)
                value = anti_selfsimilar_hash(px + extra_idx, py + extra_idx, iter_count)
                if value not in covered:
                    covered.add(value)
                    # Store in extended grid with unique virtual coords
                    vx = extra_idx + px
                    vy = extra_idx + py
                    grid[(vx, vy)] = value
                    value_to_positions[value].append((vx, vy))
        extra_idx += width
    
    # Gap-fill any remaining
    if len(covered) < 65536:
        missing = [v for v in range(65536) if v not in covered]
        print(f"  Gap-filling {len(missing)} missing values...")
        
        # Find duplicate positions (values that appear more than once)
        dupes = [(v, positions) for v, positions in value_to_positions.items() 
                 if len(positions) > 1]
        
        gap_idx = 0
        for v, positions in dupes:
            if gap_idx >= len(missing):
                break
            # Reassign last position to the missing value
            pos = positions.pop()
            old_val = grid[pos]
            new_val = missing[gap_idx]
            grid[pos] = new_val
            value_to_positions[new_val].append(pos)
            covered.add(new_val)
            gap_idx += 1
    
    print(f"  Final coverage: {len(covered)}/65536 ✅")
    
    return grid, value_to_positions

# ============================================================
# Step 2: Encode file into registry numbers
# ============================================================

def encode_file(filepath, value_to_positions):
    """Read file, split into 16-bit chunks, look up in registry."""
    data = open(filepath, 'rb').read()
    assert len(data) == 64, f"Expected 64 bytes, got {len(data)}"
    
    # Pad to even length if needed
    if len(data) % 2:
        data += b'\x00'
    
    chunks = []
    for i in range(0, len(data), 2):
        val = struct.unpack('<H', data[i:i+2])[0]
        chunks.append(val)
    
    print(f"\n64-byte file → {len(chunks)} registry lookups")
    
    # For each chunk, find candidate positions
    candidates = []
    for i, val in enumerate(chunks):
        positions = value_to_positions.get(val, [])
        candidates.append((val, positions))
        if i < 5:
            print(f"  Chunk {i:2d}: value={val:5d} ({val:016b}) → {len(positions)} position(s)")
    
    all_found = all(len(c[1]) > 0 for c in candidates)
    print(f"  All values found: {'✅' if all_found else '❌'}")
    
    total_candidates = sum(len(c[1]) for c in candidates)
    avg = total_candidates / len(candidates)
    print(f"  Average candidates per value: {avg:.1f}")
    
    return chunks, candidates

# ============================================================
# Step 3: Search for geometric patterns
# ============================================================

def try_linear_path(candidates, grid):
    """
    Try to find a linear path: start at (x0,y0), step by (dx,dy) each time.
    For each of the 32 values, check if the required value appears at position
    (x0 + i*dx, y0 + i*dy) for some (x0,y0,dx,dy).
    """
    print("\n--- Searching for LINEAR patterns (x0,y0,dx,dy) ---")
    
    n = len(candidates)
    best = None
    best_matches = 0
    tested = 0
    
    # For each possible starting position of value 0
    first_val = candidates[0][0]
    first_positions = candidates[0][1]
    
    # Only use base grid positions (< 256)
    base_first = [(x, y) for x, y in first_positions if x < 256 and y < 256]
    
    if not base_first:
        print("  First value not found in base 256×256 grid")
        return None
    
    print(f"  First value {first_val} has {len(base_first)} base positions")
    
    # Try different step sizes
    steps_to_try = []
    for dx in range(-16, 17):
        for dy in range(-16, 17):
            if dx == 0 and dy == 0:
                continue
            steps_to_try.append((dx, dy))
    
    for x0, y0 in base_first:
        for dx, dy in steps_to_try:
            tested += 1
            matches = 0
            valid = True
            
            for i in range(n):
                px = (x0 + i * dx) % 256
                py = (y0 + i * dy) % 256
                
                if (px, py) not in grid:
                    valid = False
                    break
                
                if grid[(px, py)] == candidates[i][0]:
                    matches += 1
                else:
                    # This position doesn't have the right value
                    if matches < 2:
                        break  # Early exit
            
            if matches > best_matches:
                best_matches = matches
                best = (x0, y0, dx, dy)
                if matches >= 5:
                    print(f"  Found {matches}/{n} matches: start=({x0},{y0}) step=({dx},{dy})")
            
            if matches == n:
                print(f"  🔥 PERFECT MATCH: start=({x0},{y0}) step=({dx},{dy})")
                return best
    
    print(f"  Tested {tested:,} linear paths")
    print(f"  Best: {best_matches}/{n} matches at start=({best[0]},{best[1]}) step=({best[2]},{best[3]})")
    return best if best_matches > 3 else None


def try_quadratic_path(candidates, grid):
    """
    Try quadratic paths: x(i) = x0 + dx*i + ax*i², y(i) = y0 + dy*i + ay*i²
    """
    print("\n--- Searching for QUADRATIC patterns ---")
    
    n = len(candidates)
    best = None
    best_matches = 0
    
    first_positions = [(x, y) for x, y in candidates[0][1] if x < 256 and y < 256]
    
    for x0, y0 in first_positions[:50]:  # Limit starting positions
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                for ax in range(-2, 3):
                    for ay in range(-2, 3):
                        if dx == 0 and dy == 0 and ax == 0 and ay == 0:
                            continue
                        
                        matches = 0
                        for i in range(n):
                            px = (x0 + dx*i + ax*i*i) % 256
                            py = (y0 + dy*i + ay*i*i) % 256
                            
                            if (px, py) in grid and grid[(px, py)] == candidates[i][0]:
                                matches += 1
                        
                        if matches > best_matches:
                            best_matches = matches
                            best = (x0, y0, dx, dy, ax, ay)
                            if matches >= 5:
                                print(f"  Found {matches}/{n}: start=({x0},{y0}) step=({dx},{dy}) accel=({ax},{ay})")
                        
                        if matches == n:
                            print(f"  🔥 PERFECT MATCH!")
                            return best
    
    print(f"  Best: {best_matches}/{n} matches")
    return best if best_matches > 3 else None


def analyze_storage(best_pattern, pattern_type, original_size=64):
    """Calculate storage needed for the pattern description."""
    if best_pattern is None:
        print("\n  No pattern found — cannot compress")
        return
    
    if pattern_type == 'linear':
        x0, y0, dx, dy = best_pattern
        # Storage: 2 bytes (x0,y0 as single bytes) + 2 bytes (dx,dy as signed bytes)
        storage = 4
        print(f"\n  LINEAR pattern storage:")
        print(f"    x0={x0}, y0={y0}: 2 bytes")
        print(f"    dx={dx}, dy={dy}: 2 bytes")
        print(f"    Total: {storage} bytes")
        print(f"    Original: {original_size} bytes")
        print(f"    Ratio: {storage}/{original_size} = {storage/original_size:.1%}")
    
    elif pattern_type == 'quadratic':
        x0, y0, dx, dy, ax, ay = best_pattern
        storage = 6
        print(f"\n  QUADRATIC pattern storage:")
        print(f"    x0={x0}, y0={y0}: 2 bytes")
        print(f"    dx={dx}, dy={dy}: 2 bytes")
        print(f"    ax={ax}, ay={ay}: 2 bytes")
        print(f"    Total: {storage} bytes")
        print(f"    Original: {original_size} bytes")
        print(f"    Ratio: {storage}/{original_size} = {storage/original_size:.1%}")


# ============================================================
# Main
# ============================================================

def main():
    # Create a test 64-byte file
    test_file = '/tmp/test_64bytes.bin'
    
    # Use a deterministic test pattern
    test_data = bytes(range(64))  # 0x00-0x3F
    with open(test_file, 'wb') as f:
        f.write(test_data)
    
    print("=== WIETSE FRACTAL COMPRESSION — Phase 3 PoC ===")
    print(f"Test file: {test_file} ({len(test_data)} bytes)")
    print(f"Content: {test_data.hex()[:40]}...")
    
    # Build registry
    grid, value_to_positions = build_registry()
    
    # Encode file
    chunks, candidates = encode_file(test_file, value_to_positions)
    
    # Show the 32 values we need to find
    print(f"\nTarget sequence ({len(chunks)} values):")
    for i in range(0, len(chunks), 8):
        row = chunks[i:i+8]
        print(f"  [{i:2d}-{i+len(row)-1:2d}]: {row}")
    
    # Count how many candidates per position
    print(f"\nCandidate positions per value:")
    for i, (val, positions) in enumerate(candidates):
        base_pos = [(x,y) for x,y in positions if x < 256 and y < 256]
        print(f"  [{i:2d}] value={val:5d}: {len(base_pos)} base grid positions, {len(positions)} total")
    
    # Search for patterns
    linear = try_linear_path(candidates, grid)
    
    if linear:
        analyze_storage(linear, 'linear')
    
    quad = try_quadratic_path(candidates, grid)
    
    if quad:
        analyze_storage(quad, 'quadratic')
    
    if not linear and not quad:
        print("\n⚠️  No geometric pattern found for this file content.")
        print("This means the specific sequence of values doesn't align")
        print("along any simple geometric path on the fractal grid.")
        print("\nNext approaches to try:")
        print("1. Larger fractal (more positions per value = more candidates)")
        print("2. Multiple fractal zoom levels (each produces different values)")
        print("3. Relaxed patterns (piecewise linear, spline paths)")
        print("4. Different file content (some sequences may be easier)")

if __name__ == '__main__':
    main()
