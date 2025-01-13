use rand::Rng;
use std::fs::File;
use std::io::{self, Write};
use std::collections::HashMap;
use std::collections::HashSet;
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

    let centroid: u8 = get_cluster(&centroids, img_vec);
    println!("Nearest centroid: {}   Prediction: {}", centroid, map_centroid_to_class(centroid));
    return;
  }

  let mut centroids: Vec<Vec<u8>> = Vec::new();
  match data_loader::load_model() {
      Ok(value) => {
        centroids = value.clone();
      }
      Err(e) => println!("Error: {}", e),
  }

  let train_dataset: Vec<(Vec<u8>, u8)> = data_loader::load_train_dataset();

  // let silhouette_coeff: f32 = compute_silhouette_coeff(&train_dataset, &centroids);
  // println!("Silhouette coefficient: {}", silhouette_coeff);

  println!("Calculating Rand Index...");
  let mut true_clusters: Vec<u8> = train_dataset.iter().map(|(_, cluster)| *cluster).collect();
  let mut predicted_clusters: Vec<u8> = train_dataset.iter().map(|(image_vec, _)| get_cluster(&centroids, image_vec.clone())).collect();
  let rand_index: f32 = rand_index(&true_clusters, &predicted_clusters);
  println!("Rand Index: {}", rand_index);
}

fn map_centroid_to_class(centroid: u8) -> u8 {
  match centroid {
    0..=3 => 0,
    4..=7 => 1,
    8..=11 => 2,
    12..=15 => 3,
    16..=19 => 4,
    20..=23 => 5,
    24..=27 => 6,
    28..=31 => 7,
    32..=36 => 8,
    37..=42 => 9,
    _ => 255,
  }
}

fn get_cluster(centroids: &Vec<Vec<u8>>, image: Vec<u8>) -> u8 {
  let mut min_dist: f32 = f32::INFINITY;
  let mut class: u8 = 255;

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

fn compute_silhouette_coeff(images: &Vec<(Vec<u8>, u8)>, centroids: &Vec<Vec<u8>>) -> f32 {
  let mut silhouette_coeff: f32 = 0.0;
  let mut cluster_sizes: Vec<u32> = vec![0; centroids.len()];
  let mut a: Vec<f32> = vec![0.0; images.len()];
  let mut b: Vec<f32> = vec![0.0; images.len()];

  let mut images: Vec<(Vec<u8>, u8)> = images.clone();
  images.iter_mut().for_each(|(image_vec, cluster)| *cluster = get_cluster(&centroids, image_vec.clone()));
  let mut cluster_images: HashMap<u8, Vec<Vec<u8>>> = HashMap::new();
  for (image_vec, cluster) in images.iter() {
    cluster_images.entry(*cluster).or_insert(Vec::new()).push(image_vec.clone());
  }

  for (i, (image_vec, cluster)) in images.iter().enumerate() {
    let mut min_dist: f32 = f32::INFINITY;
    for (centroid_cluster, centroid) in centroids.iter().enumerate() {
        if (centroid_cluster as u8) == *cluster {
            continue;
        }

        let dist: f32 = distance(&image_vec, &centroid);
        if dist < min_dist {
            min_dist = dist;
        }
    }
    b[i] = min_dist;
  }

  for (i, (image_vec, cluster)) in images.iter().enumerate() {
    let mut sum: f32 = 0.0;
    let mut count: u32 = 0;
    for image_vec2 in cluster_images.get(cluster).unwrap() {
        sum += distance(&image_vec, &image_vec2);
        count += 1;
    }

    a[i] = sum / count as f32;
  }

  for i in 0..images.len() {
    silhouette_coeff += (b[i] - a[i]) / a[i].max(b[i]);
  }

  silhouette_coeff / images.len() as f32
}

fn rand_index(true_clusters: &Vec<u8>, predicted_clusters: &Vec<u8>) -> f32 {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    let n = true_clusters.len();

    for i in 0..n {
        for j in i + 1..n {
            let same_true_cluster = true_clusters[i] == true_clusters[j];
            let same_predicted_cluster = predicted_clusters[i] == predicted_clusters[j];

            if same_true_cluster && same_predicted_cluster {
                tp += 1; // True Positive
            } else if !same_true_cluster && !same_predicted_cluster {
                tn += 1; // True Negative
            } else if same_true_cluster && !same_predicted_cluster {
                fp += 1; // False Positive
            } else {
                fn_count += 1; // False Negative
            }
        }

        if i % 100 == 0 {
            println!("Iteration: {} / 60000", i);
        }
    }

    // Calculate the Rand Index
    let rand_index = (tp + tn) as f32 / (tp + tn + fp + fn_count) as f32;
    rand_index
}

fn train(images: &mut Vec<(Vec<u8>, u8)>) {
  let mut centroids: Vec<Vec<u8>> = Vec::new();
  for _ in 0..=9 {
    // set random datapoint as centroid
    centroids.push(data_loader::img_to_vec(select_random_file()));
  }

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

fn select_random_file() -> String {
  let mut rng = rand::thread_rng();
  let random_number: usize = rng.gen_range(0..=9);
  let mut directory: String = format!("MNIST/train/{}/", random_number);
  let mut path: String = data_loader::get_random_img(&directory);
  println!("Random centroid: {}", path);
  path
}
