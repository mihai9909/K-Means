use rand::Rng;
use std::fs::File;
use std::io::{self, Write};
mod data_loader;

fn main() {
  let mut images: Vec<(Vec<u8>, u8)> = data_loader::load_train_dataset(); // vector of tuples containing the image as vector and the class of the image (default 10)

  let mut centroids: Vec<Vec<u8>> = generate_random_centroids();

  let mut made_changes: bool = true;
  while made_changes {
    made_changes = false;
    for i in 0..images.len() {
      // compute distance to each centroid
      let mut min_dist: f32 = f32::INFINITY;
      let prev_cluster = images[i].1;
      for j in 0..centroids.len() {
        let dist: f32 = distance(&images[i].0, &centroids[j]);
        // assign cluster to distance to minimum centroid
        if dist < min_dist {
          min_dist = dist;
          images[i].1 = j as u8;
        }
      }

      if prev_cluster != images[i].1 { // if cluster changed
        made_changes = true;
      }
    }

    // compute average
    let mut new_centroids: Vec<Vec<u32>> = vec![vec![0; 784]; 10];
    let mut cluster_sizes: Vec<u32> = vec![0; 10];
    for img in &images {
      match img.1 {
        0=> { sum(&mut new_centroids[0], &img.0); cluster_sizes[0]+=1; },
        1=> { sum(&mut new_centroids[1], &img.0); cluster_sizes[1]+=1; },
        2=> { sum(&mut new_centroids[2], &img.0); cluster_sizes[2]+=1; },
        3=> { sum(&mut new_centroids[3], &img.0); cluster_sizes[3]+=1; },
        4=> { sum(&mut new_centroids[4], &img.0); cluster_sizes[4]+=1; },
        5=> { sum(&mut new_centroids[5], &img.0); cluster_sizes[5]+=1; },
        6=> { sum(&mut new_centroids[6], &img.0); cluster_sizes[6]+=1; },
        7=> { sum(&mut new_centroids[7], &img.0); cluster_sizes[7]+=1; },
        8=> { sum(&mut new_centroids[8], &img.0); cluster_sizes[8]+=1; },
        9=> { sum(&mut new_centroids[9], &img.0); cluster_sizes[9]+=1; },
        _=> panic!("Invalid cluster!"), 
      }
    }

    for i in 0..=9 {
      if cluster_sizes[i] == 0 {
        continue;
      }

      div(&mut new_centroids[i], cluster_sizes[i]);
      for j in 0..new_centroids[i].len() {
        centroids[i][j] = new_centroids[i][j] as u8;
      }
    }
    println!("Cluster sizes: {:?}", cluster_sizes);
  }

  save_model(&centroids);
}

pub fn save_model(data: &Vec<Vec<u8>>) -> io::Result<()> {
  let mut file = File::create("model.txt")?;

  for centroid in data {
      writeln!(file, "{:?}", centroid)?;
  }

  Ok(())
}

fn sum(vec1: &mut Vec<u32>, vec2: &Vec<u8>) {
  for i in 0..vec1.len() {
    vec1[i] = vec1[i] + vec2[i] as u32;
  }
}

fn div(vec1: &mut Vec<u32>, denominator: u32 ) {
  for i in 0..vec1.len() {
    vec1[i] = vec1[i] as u32 / denominator;
  }
}

fn distance(a: &Vec<u8>, b: &Vec<u8>) -> f32 { // euclidian distance between two vectors
  let mut sum: f32 = 0.0;
  for i in 0..a.len() {
    let diff: f32 = (a[i] as f32 - b[i] as f32).into();
    sum += (diff)*(diff);
  };
  f32::sqrt(sum)
}

fn generate_random_vec() -> Vec<u8> {
  let mut rng = rand::thread_rng();
  (0..784).map(|_| rng.gen_range(0..=255)).collect()
}

fn generate_random_centroids() -> Vec<Vec<u8>> {
  let mut centroids: Vec<Vec<u8>> = Vec::new();
  for _ in 0..=9 {
    centroids.push(generate_random_vec());
  }

  centroids
}
