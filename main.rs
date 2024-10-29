use rand::Rng;
use std::fs::File;
use std::io::{self, Write};
mod data_loader;

fn main() {
  // let mut images: Vec<(Vec<u8>, u8)> = data_loader::load_train_dataset(); // vector of tuples containing the image as vector and the class of the image (default 10)

  //train(&mut images);

  let mut centroids: Vec<Vec<u8>> = Vec::new();
  match data_loader::load_model() {
      Ok(value) => {
        centroids = value.clone();
      }
      Err(e) => println!("Error: {}", e),
  }

  let image: Vec<u8> = data_loader::img_to_vec("MNIST/train/6/13.png".to_string());

  let class: u8 = classify(centroids, image);

  println!("class: {}", class);
}

fn classify(centroids: Vec<Vec<u8>>, image: Vec<u8>) -> u8 {
  let mut min_dist: f32 = f32::INFINITY;
  let mut class: u8 = 10;

  for j in 0..centroids.len() {
    let dist: f32 = distance(&image, &centroids[j]);
    // assign cluster to distance to minimum centroid
    if dist < min_dist {
      min_dist = dist;
      class = j as u8;
    }
  }

  class
}

fn train(images: &mut Vec<(Vec<u8>, u8)>) {
  let mut centroids: Vec<Vec<u8>> = generate_random_centroids();

  let mut made_changes: bool = true;
  while made_changes {
    made_changes = false;
    for (image_vec, image_cluster) in images.iter_mut() {
      // compute distance to each centroid
      let mut min_dist: f32 = f32::INFINITY;
      let prev_cluster = *image_cluster;
      for j in 0..centroids.len() {
        let dist: f32 = distance(&image_vec, &centroids[j]);
        // assign cluster to distance to minimum centroid
        if dist < min_dist {
          min_dist = dist;
          *image_cluster = j as u8;
        }
      }

      if prev_cluster != *image_cluster { // if cluster changed
        made_changes = true;
      }
    }

    // compute average
    let mut new_centroids: Vec<Vec<u32>> = vec![vec![0; 784]; 10];
    let mut cluster_sizes: Vec<u32> = vec![0; 10];
    for (image_vec, image_cluster) in images.iter_mut() {
      match *image_cluster {
        0=> { sum(&mut new_centroids[0], &image_vec); cluster_sizes[0]+=1; },
        1=> { sum(&mut new_centroids[1], &image_vec); cluster_sizes[1]+=1; },
        2=> { sum(&mut new_centroids[2], &image_vec); cluster_sizes[2]+=1; },
        3=> { sum(&mut new_centroids[3], &image_vec); cluster_sizes[3]+=1; },
        4=> { sum(&mut new_centroids[4], &image_vec); cluster_sizes[4]+=1; },
        5=> { sum(&mut new_centroids[5], &image_vec); cluster_sizes[5]+=1; },
        6=> { sum(&mut new_centroids[6], &image_vec); cluster_sizes[6]+=1; },
        7=> { sum(&mut new_centroids[7], &image_vec); cluster_sizes[7]+=1; },
        8=> { sum(&mut new_centroids[8], &image_vec); cluster_sizes[8]+=1; },
        9=> { sum(&mut new_centroids[9], &image_vec); cluster_sizes[9]+=1; },
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
