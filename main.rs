use rand::Rng;
mod data_loader;

fn main() {
  let images: Vec<(Vec<u8>, u8)> = data_loader::load_train_dataset(); // vector of tuples containing the image as vector and the class of the image (default 10)
  let distance: f32 = distance(&images[0].0, &images[40000].0);
  println!("distance: {}", distance);

  let mut centroids: Vec<Vec<u8>> = generate_random_centroids();

  println!("Random numbers: {:?}", centroids[9]);

}

fn distance(a: &Vec<u8>, b: &Vec<u8>) -> f32 {
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
