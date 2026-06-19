/// Wietse Fractal Compression — Viability Test
///
/// Question: At what string length can we reliably find a target
/// sequence on the fractal grid along a geometric path?
///
/// Test: For lengths 1, 2, 3, 4, ... try to find the target values
/// along ANY linear path on a fractal grid. Report success/failure.
///
/// Grids tested: 256×256, 1024×1024, 4096×4096
/// No compression goals — pure discovery.

use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

const MAX_ITER: u32 = 65536;

// ─── Mandelbrot with fast interior skip ────────────────────────────────────

#[inline(always)]
fn mandelbrot(cx: f64, cy: f64) -> u32 {
    let x1 = cx - 0.25;
    let cy2 = cy * cy;
    let q = x1 * x1 + cy2;
    if q * (q + x1) <= 0.25 * cy2 {
        return MAX_ITER;
    }
    let x2 = cx + 1.0;
    if x2 * x2 + cy2 <= 0.0625 {
        return MAX_ITER;
    }
    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    for i in 0..MAX_ITER {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > 4.0 { return i; }
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
    }
    MAX_ITER
}

// ─── Anti-self-similar hash ────────────────────────────────────────────────

#[inline(always)]
fn hash_point(x: u64, y: u64, iter: u32) -> u16 {
    let mut h = iter as u64;
    h = h.wrapping_mul(2_654_435_761);
    h ^= x.wrapping_mul(0x9E3779B97F4A7C15);
    h = h.rotate_left(17);
    h ^= y.wrapping_mul(0x517CC1B727220A95);
    h = h.wrapping_mul(0x6C62272E07BB0142);
    h = h.rotate_left(31);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CEB9FE1A85EC53);
    h ^= h >> 33;
    (h & 0xFFFF) as u16
}

// ─── Grid generation (parallel by rows) ────────────────────────────────────

fn generate_grid(w: usize, h: usize, cx: f64, cy: f64, span: f64) -> Vec<u16> {
    let x_min = cx - span;
    let y_min = cy - span;
    let sx = 2.0 * span / w as f64;
    let sy = 2.0 * span / h as f64;

    let mut grid = vec![0u16; w * h];

    // Parallel by row chunks
    grid.par_chunks_mut(w)
        .enumerate()
        .for_each(|(py, row)| {
            let fcy = y_min + py as f64 * sy;
            for px in 0..w {
                let fcx = x_min + px as f64 * sx;
                let iter = mandelbrot(fcx, fcy);
                row[px] = hash_point(px as u64, py as u64, iter);
            }
        });

    grid
}

// ─── Sequence search ───────────────────────────────────────────────────────

/// Build reverse index: value → list of (x, y) positions
fn build_index(grid: &[u16], w: usize, h: usize) -> HashMap<u16, Vec<(usize, usize)>> {
    let mut idx: HashMap<u16, Vec<(usize, usize)>> = HashMap::new();
    for py in 0..h {
        for px in 0..w {
            let v = grid[py * w + px];
            idx.entry(v).or_default().push((px, py));
        }
    }
    idx
}

/// Try to find target[0..len] along a linear path on the grid.
/// Returns Some((sx, sy, dx, dy)) if found, None otherwise.
fn find_sequence(
    grid: &[u16],
    w: usize,
    h: usize,
    index: &HashMap<u16, Vec<(usize, usize)>>,
    target: &[u16],
    max_step: i32,
) -> Option<(usize, usize, i32, i32, usize)> {
    let n = target.len();
    if n == 0 { return None; }

    // Get all positions where target[0] appears
    let starts = match index.get(&target[0]) {
        Some(positions) => positions.clone(),
        None => return None,
    };

    if n == 1 {
        let (sx, sy) = starts[0];
        return Some((sx, sy, 0, 0, 1));
    }

    let mut global_best = 0usize;
    let mut global_result = None;

    // For each starting position, try all step sizes
    for &(sx, sy) in &starts {
        for dx in -max_step..=max_step {
            for dy in -max_step..=max_step {
                if dx == 0 && dy == 0 { continue; }

                let mut matches = 1usize; // target[0] already matches
                let mut all_match = true;

                for i in 1..n {
                    let px = ((sx as i32 + i as i32 * dx).rem_euclid(w as i32)) as usize;
                    let py = ((sy as i32 + i as i32 * dy).rem_euclid(h as i32)) as usize;

                    if grid[py * w + px] == target[i] {
                        matches += 1;
                    } else {
                        all_match = false;
                        // For short sequences, break early if any miss
                        if n <= 8 { break; }
                    }
                }

                if all_match && matches == n {
                    return Some((sx, sy, dx, dy, matches));
                }

                if matches > global_best {
                    global_best = matches;
                    global_result = Some((sx, sy, dx, dy, matches));
                }
            }
        }
    }

    global_result
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("=== WIETSE FRACTAL — Sequence Viability Test ===");
    println!("max_iter: {}", MAX_ITER);
    println!();

    // Create target data from a deterministic 64-byte file
    let file_data: Vec<u8> = (0u8..64).collect();
    let all_targets: Vec<u16> = (0..32)
        .map(|i| u16::from_le_bytes([file_data[i * 2], file_data[i * 2 + 1]]))
        .collect();

    println!("Full target (32 values): {:?}", &all_targets[..8]);
    println!();

    // Test with increasing grid sizes
    let grid_configs: Vec<(usize, f64, f64, f64, i32)> = vec![
        // (size, cx, cy, span, max_step)
        (1024,  -0.75, 0.0, 1.5, 64),
        (4096,  -0.75, 0.0, 1.5, 64),
        (8192,  -0.75, 0.0, 1.5, 32),
        (16384, -0.75, 0.0, 1.5, 16),  // 512MB, smaller steps for speed
    ];

    for &(size, cx, cy, span, max_step) in &grid_configs {
        let total_points = size * size;
        let mem_mb = total_points * 2 / 1024 / 1024;

        println!("╔══════════════════════════════════════════════╗");
        println!("║  Grid: {}×{} = {} points ({} MB)  ", size, size, total_points, mem_mb);
        println!("║  Viewport: ({}, {}) span={}        ", cx, cy, span);
        println!("║  Step range: ±{}                   ", max_step);
        println!("╚══════════════════════════════════════════════╝");

        let t0 = Instant::now();
        let grid = generate_grid(size, size, cx, cy, span);
        let gen_time = t0.elapsed();
        println!("  Grid generated in {:.2}s", gen_time.as_secs_f64());

        // Stats
        let unique: std::collections::HashSet<u16> = grid.iter().copied().collect();
        let avg_copies = total_points as f64 / unique.len() as f64;
        println!("  Unique values: {} / 65536 ({:.1} copies avg)",
                 unique.len(), avg_copies);

        // Build reverse index
        let t1 = Instant::now();
        let index = build_index(&grid, size, size);
        println!("  Index built in {:.2}s", t1.elapsed().as_secs_f64());

        // Check: how many of our target values exist in this grid?
        let targets_present = all_targets.iter()
            .filter(|v| index.contains_key(v))
            .count();
        println!("  Target values present: {}/{}",
                 targets_present, all_targets.len());

        // Show copies per target value
        println!("  Copies per target value:");
        for (i, v) in all_targets.iter().enumerate().take(8) {
            let copies = index.get(v).map(|p| p.len()).unwrap_or(0);
            print!("    [{:2}]={:5}→{:3} copies", i, v, copies);
            if i % 2 == 1 { println!(); }
        }
        if all_targets.len() > 8 { println!("    ..."); }

        // Test increasing sequence lengths
        println!("\n  Sequence search (length → result):");
        println!("  {:<6} {:<12} {:<30} {:<10}", "Len", "Found?", "Path", "Time");
        println!("  {}", "─".repeat(60));

        let mut max_viable = 0;

        for seq_len in 1..=all_targets.len().min(16) {
            let target_slice = &all_targets[..seq_len];

            // Check if all values exist in grid
            let all_present = target_slice.iter().all(|v| index.contains_key(v));
            if !all_present {
                println!("  {:<6} {:<12} {:<30} {:<10}",
                         seq_len, "SKIP", "target value(s) not in grid", "—");
                continue;
            }

            let t2 = Instant::now();
            let result = find_sequence(&grid, size, size, &index, target_slice, max_step);
            let search_time = t2.elapsed();

            match result {
                Some((sx, sy, dx, dy, matches)) if matches == seq_len => {
                    max_viable = seq_len;
                    println!("  {:<6} {:<12} ({},{}) step({},{})        {:.3}s",
                             seq_len, "✅ FOUND", sx, sy, dx, dy,
                             search_time.as_secs_f64());

                    // Verify
                    let mut ok = true;
                    for i in 0..seq_len {
                        let px = ((sx as i32 + i as i32 * dx).rem_euclid(size as i32)) as usize;
                        let py = ((sy as i32 + i as i32 * dy).rem_euclid(size as i32)) as usize;
                        if grid[py * size + px] != target_slice[i] {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        println!("         ⚠️  VERIFICATION FAILED");
                    }
                }
                Some((_, _, _, _, matches)) => {
                    println!("  {:<6} {:<12} best {}/{} matches           {:.3}s",
                             seq_len, "❌ PARTIAL", matches, seq_len,
                             search_time.as_secs_f64());
                }
                None => {
                    println!("  {:<6} {:<12} {:<30} {:.3}s",
                             seq_len, "❌ NONE", "no starting position", 
                             search_time.as_secs_f64());
                }
            }

            // If search took more than 60s per length, stop
            if search_time.as_secs() > 60 {
                println!("  (stopping — search too slow at this grid size)");
                break;
            }
        }

        println!("\n  ✦ Max viable sequence length on {}×{}: {}", size, size, max_viable);
        println!("  ✦ Total grid time: {:.1}s", t0.elapsed().as_secs_f64());
        println!();

        // Theoretical analysis for this grid
        let starts_per_value = avg_copies;
        let steps_tested = (2 * max_step + 1) * (2 * max_step + 1) - 1;
        let p_match: f64 = 1.0 / 65536.0;
        println!("  Theory for this grid:");
        println!("    Starts per target value: {:.1}", starts_per_value);
        println!("    Steps tested per start:  {}", steps_tested);
        println!("    P(next value matches):   1/{}", 65536.0 / (unique.len() as f64 / 65536.0 * unique.len() as f64).min(65536.0));
        let total_trials = starts_per_value * steps_tested as f64;
        println!("    Total path trials:       {:.0}", total_trials);

        // Expected max consecutive matches
        // For each trial, P(k consecutive) = (1/65536)^(k-1)
        // With N trials, expected max k where N * p^(k-1) >= 1
        let n = total_trials;
        if n > 0.0 {
            let mut k = 1;
            while n * p_match.powi(k - 1) >= 1.0 {
                k += 1;
            }
            println!("    Expected max consecutive: ~{}", k);
        }

        println!("\n{}", "═".repeat(50));
        println!();
    }

    println!("=== EXPERIMENT COMPLETE ===");
    println!();
    println!("Key question: At what grid size does sequence length N become viable?");
    println!("The relationship: grid_area × steps / 65536^(N-1) ≥ 1");
    println!();
    println!("For length 2:  need ~{:.0} grid area (with ±64 steps)", 65536.0 / (129.0 * 129.0 - 1.0));
    println!("For length 3:  need ~{:.0} grid area", 65536.0f64.powi(2) / (129.0 * 129.0 - 1.0));
    println!("For length 4:  need ~{:.0} grid area", 65536.0f64.powi(3) / (129.0 * 129.0 - 1.0));
    println!("For length 5:  need ~{:.0} grid area", 65536.0f64.powi(4) / (129.0 * 129.0 - 1.0));
}
