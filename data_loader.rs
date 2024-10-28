use image::ImageReader;
use std::fs;

fn img_to_vec(path: String) -> Vec<u8> {
  let img = ImageReader::open(path)
  .expect("Failed to open image")
  .decode()
  .expect("Failed to decode image");

  let img_gray = img.to_luma8();
  img_gray.into_raw()
}

pub fn load_train_dataset() -> Vec<Vec<u8>> {
  let mut all_images: Vec<Vec<u8>> = Vec::new();
  for i in 0..10 {
    let path = format!("MNIST/{}/", i);
    let images: Vec<Vec<u8>> = load_images(&path);
    all_images.extend(images);
  }
  println!("Number of images: {}", all_images.len());
  all_images
}

pub fn load_images(directory_path: &str) -> Vec<Vec<u8>> {
  let mut images: Vec<Vec<u8>> = vec![];

  match fs::read_dir(directory_path) {
    Ok(entries) => {
      for entry in entries  {
        match entry {
          Ok(entry) => {
            let file_name = entry.file_name();
            // println!("{:?}", file_name);
            let mut path_to_file = directory_path.to_string(); // Start with a mutable String
            path_to_file.push_str(&file_name.to_string_lossy()); // Append the file name
            images.push(img_to_vec(path_to_file));
          }
          Err(e) => eprintln!("Error reading entry: {}", e),
        }
      }
    }
    Err(e) => eprintln!("Failed to read directory: {}", e),
  }

  images
}

fn print_pixel_data(pixels: Vec<u8>) {
  for (i, pixel) in pixels.iter().enumerate() {
    print!("{} ", pixel);
    if i % 28 == 0 {
      println!();
    }
  }
}

fn print_vec_len(pixels: Vec<u8>) {
  println!("Grayscale pixel data length: {}", pixels.len());
}
