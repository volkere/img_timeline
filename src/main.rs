use std::fs;
use std::path::Path;
use std::collections::HashMap;
use std::process::Command;
use serde_json::json;
use opencv::{prelude::*, core, imgcodecs, imgproc, objdetect};

fn detect_faces(image_path: &str) -> Vec<(i32, i32, i32, i32)> {
    let face_cascade = objdetect::CascadeClassifier::new("haarcascade_frontalface_default.xml").unwrap();
    let img = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR).unwrap();
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0).unwrap();
    
    let mut faces = core::Vector::new();
    face_cascade.detect_multi_scale(&gray, &mut faces, 1.1, 5, 0, core::Size::new(30, 30), core::Size::new(300, 300)).unwrap();
    
    faces.iter().map(|r| (r.x, r.y, r.width, r.height)).collect()
}

fn estimate_age(image_path: &str) -> Vec<i32> {
    let output = Command::new("python3")
        .arg("age_estimation.py")
        .arg(image_path)
        .output()
        .expect("Failed to execute Python script");
    
    let result = String::from_utf8_lossy(&output.stdout);
    result.trim().split_whitespace().filter_map(|x| x.parse::<i32>().ok()).collect()
}

fn tag_faces(image_path: &str) -> Vec<HashMap<String, String>> {
    let faces = detect_faces(image_path);
    let ages = estimate_age(image_path);
    
    let mut face_data: Vec<(i32, (i32, i32, i32, i32))> = ages.into_iter().zip(faces.into_iter()).collect();
    face_data.sort_by(|a, b| a.0.cmp(&b.0));
    
    let mut tagged_faces = Vec::new();
    for (index, &(age, (x, y, w, h))) in face_data.iter().enumerate() {
        let mut face_info = HashMap::new();
        face_info.insert("Tag".to_string(), format!("{}", index + 1));
        face_info.insert("Alter".to_string(), format!("{}", age));
        face_info.insert("Koordinaten".to_string(), format!("({}, {}, {}, {})", x, y, w, h));
        tagged_faces.push(face_info);
    }
    
    tagged_faces
}

fn save_face_data(face_data: &Vec<HashMap<String, String>>, output_file: &str) {
    let json_data = json!(face_data).to_string();
    fs::write(format!("{}.json", output_file), json_data).unwrap();
}

fn annotate_image(image_path: &str, face_data: &Vec<HashMap<String, String>>) {
    let mut img = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR).unwrap();
    for face in face_data {
        if let Some(coords) = face.get("Koordinaten") {
            let parts: Vec<i32> = coords
                .trim_matches(&['(', ')'] as &[_])
                .split(", ")
                .filter_map(|x| x.parse::<i32>().ok())
                .collect();
            if parts.len() == 4 {
                let rect = core::Rect::new(parts[0], parts[1], parts[2], parts[3]);
                imgproc::rectangle(&mut img, rect, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0).unwrap();
            }
        }
    }
    imgcodecs::imwrite("annotated_output.jpg", &img, &core::Vector::new()).unwrap();
}

fn main() {
    let image_path = "input.jpg";
    let output_file = "face_tags";
    
    let face_data = tag_faces(image_path);
    save_face_data(&face_data, output_file);
    annotate_image(image_path, &face_data);
}

