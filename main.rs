use rand::Rng;
use std::fs::File;
use std::io::{self, Write};
mod data_loader;
use std::env;

fn main() {
  let args: Vec<String> = env::args().collect();

  if args.len() == 2 && "train".eq(&args[1]) {
    let mut images: Vec<(Vec<u8>, u8)> = data_loader::load_train_dataset(); // vector of tuples containing the image as vector and the class of the image (default 10)

    train(&mut images);
    
    return;
  } else if args.len() == 2 {
    let mut centroids: Vec<Vec<u8>> = Vec::new();
    match data_loader::load_model() {
        Ok(value) => {
          centroids = value.clone();
        }
        Err(e) => println!("Error: {}", e),
    }
  
    let img_vec: Vec<u8> = data_loader::img_to_vec(args[1].to_string());

    let centroid: u8 = classify(&centroids, img_vec);
    println!("Prediction: {}", centroid);
    return;
  }

  let mut centroids: Vec<Vec<u8>> = Vec::new();
  match data_loader::load_model() {
      Ok(value) => {
        centroids = value.clone();
      }
      Err(e) => println!("Error: {}", e),
  }

  let test_dataset: Vec<(Vec<u8>, u8)> = data_loader::load_test_dataset();

  let mut correct_guesses: f32 = 0.0;
  for (img_vec, img_cluster) in test_dataset {
    let centroid_number: u8 = classify(&centroids, img_vec);
    if map_centroid_to_class(centroid_number) == img_cluster {
      correct_guesses += 1.0;
    }
  }

  println!("Correct answers: {}", correct_guesses);
  println!("Accuracy: {}", correct_guesses/10000.0);
}

fn map_centroid_to_class(centroid: u8) -> u8 {
  match centroid {
    0 => 0,
    1 => 1,
    2 => 2,
    3 => 3,
    4 | 5 | 6 => 4,
    7 | 8 | 9 | 10 => 5,
    11 => 6,
    12 => 7,
    13 => 8,
    14 | 15 => 9,
    _ => 255,
  }
}

fn classify(centroids: &Vec<Vec<u8>>, image: Vec<u8>) -> u8 {
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
  let mut centroids: Vec<Vec<u8>> = Vec::new();
  centroids.push(data_loader::img_to_vec("MNIST/train/0/6422.png".to_string())); // cluster 0
  centroids.push(data_loader::img_to_vec("MNIST/train/1/21002.png".to_string())); // cluster 1
  centroids.push(data_loader::img_to_vec("MNIST/train/2/23691.png".to_string())); // cluster 2
  centroids.push(data_loader::img_to_vec("MNIST/train/3/21831.png".to_string())); // cluster 3

  centroids.push(data_loader::img_to_vec("MNIST/train/4/35752.png".to_string())); // cluster 4 = img 4
  centroids.push(data_loader::img_to_vec("MNIST/train/4/150.png".to_string())); // cluster 5 = img 4
  centroids.push(data_loader::img_to_vec("MNIST/train/4/1460.png".to_string())); // cluster 6 = img 4

  centroids.push(data_loader::img_to_vec("MNIST/train/5/20601.png".to_string())); // cluster 7 = img 5
  centroids.push(data_loader::img_to_vec("MNIST/train/5/1311.png".to_string())); // cluster 8 = img 5
  centroids.push(data_loader::img_to_vec("MNIST/train/5/31982.png".to_string())); // cluster 9 = img 5
  centroids.push(data_loader::img_to_vec("MNIST/train/5/31999.png".to_string())); // cluster 10 = img 5

  centroids.push(data_loader::img_to_vec("MNIST/train/6/59848.png".to_string())); // cluster 11 = img 6
  centroids.push(data_loader::img_to_vec("MNIST/train/7/44803.png".to_string())); // cluster 12 = img 7
  centroids.push(data_loader::img_to_vec("MNIST/train/8/53464.png".to_string())); // cluster 13 = img 8

  centroids.push(data_loader::img_to_vec("MNIST/train/9/5320.png".to_string())); // cluster 14 = img 9
  centroids.push(data_loader::img_to_vec("MNIST/train/9/170.png".to_string())); // cluster 15 = img 9

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
    let mut new_centroids: Vec<Vec<u32>> = vec![vec![0; 784]; centroids.len()];
    let mut cluster_sizes: Vec<u32> = vec![0; centroids.len()];
    for (image_vec, image_cluster) in images.iter_mut() {
      sum(&mut new_centroids[*image_cluster as usize], &image_vec); cluster_sizes[*image_cluster as usize]+=1;
    }

    for i in 0..cluster_sizes.len() {
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

  let _ = save_model(&centroids);
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
