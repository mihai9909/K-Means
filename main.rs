mod data_loader;

fn main() {
  // let pixels: Vec<u8> = img_to_vec("MNIST/0/1.png");

  // print_pixel_data(pixels);

  // let path = "MNIST/0/";
  // let mut images: Vec<Vec<u8>> = load_images("MNIST/0/");
  // println!("Number of images: {}", images.len());
  // for image in images {
  //   print_vec_len(image);
  // }
  // let random_numbers: Vec<i32> = (0..length)
  //       .map(|_| rng.gen_range(1..=100)) // Generates random numbers in the range [1, 100]
  //       .collect();
  
  let images: Vec<Vec<u8>> = data_loader::load_train_dataset();
}
