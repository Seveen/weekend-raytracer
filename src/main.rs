use anyhow::Result;
use glam::{vec3, Vec3};
use std::{fmt::Write, fs, ops::Range};

trait ColorTrait {
    fn write_color(&self, buffer: &mut String) -> Result<()>;
}

impl ColorTrait for Vec3 {
    fn write_color(&self, buffer: &mut String) -> Result<()> {
        let intensity = 0.0..0.999;

        let r = (intensity.clamp(self.x) * 256.0) as u32;
        let g = (intensity.clamp(self.y) * 256.0) as u32;
        let b = (intensity.clamp(self.z) * 256.0) as u32;

        writeln!(buffer, "{r} {g} {b}")?;

        Ok(())
    }
}

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }

    pub fn color(&self, world: &dyn Hittable) -> Vec3 {
        if let Some(hit_record) = world.hit(self, 0.0..f32::INFINITY) {
            return 0.5 * (hit_record.normal + Vec3::splat(1.0));
        }

        let unit_direction = self.direction.normalize_or_zero();
        let a = 0.5 * (unit_direction.y + 1.0);

        vec3(1.0, 1.0, 1.0).lerp(vec3(0.5, 0.7, 1.0), a)
    }
}

pub struct Camera {
    focal_length: f32,
    viewport_height: f32,
    viewport_width: f32,
    image_height: u32,
    image_width: u32,
    center: Vec3,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    viewport_upper_left: Vec3,
    pixel00_location: Vec3,
    samples_per_pixel: u32,
    pixel_samples_scale: f32,
}

impl Camera {
    pub fn new(image_width: u32, aspect_ratio: f32) -> Self {
        let center = Vec3::ZERO;
        let viewport_height = 2.0;
        let focal_length = 1.0;

        let image_height = (image_width as f32 / aspect_ratio) as u32;
        let viewport_width = viewport_height * (image_width as f32 / image_height as f32);

        let viewport_u = vec3(viewport_width, 0.0, 0.0);
        let viewport_v = vec3(0.0, -viewport_height, 0.0);

        let pixel_delta_u = viewport_u / image_width as f32;
        let pixel_delta_v = viewport_v / image_height as f32;

        let viewport_upper_left = center
            - vec3(0.0, 0.0, focal_length)
            - viewport_u / 2.0
            - viewport_v / 2.0;

        let pixel00_location = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        let samples_per_pixel = 100;
        let pixel_samples_scale = 1.0 / samples_per_pixel as f32;

        Self {
            focal_length,
            viewport_height,
            viewport_width,
            image_height,
            image_width,
            center,
            pixel_delta_u,
            pixel_delta_v,
            viewport_upper_left,
            pixel00_location,
            samples_per_pixel,
            pixel_samples_scale,
        }
    }

    pub fn render<F>(&self, world: &dyn Hittable, mut on_pixel: F)
    where
        F: FnMut(u32, u32, Vec3),
    {
        for j in 0..self.image_height {
            println!("Scanlines remaining: {}", self.image_height - j);
            for i in 0..self.image_width {
                let mut pixel_color = Vec3::ZERO;
                for _sample in 0..self.samples_per_pixel {
                    let ray = self.get_ray(i, j);
                    pixel_color += ray.color(world);
                }

                on_pixel(i, j, self.pixel_samples_scale * pixel_color);
            }
        }
    }

    fn get_ray(&self, i: u32, j: u32) -> Ray {
        let offset = sample_square();
        let pixel_sample = self.pixel00_location
            + ((i as f32 + offset.x) * self.pixel_delta_u)
            + ((j as f32 + offset.y) * self.pixel_delta_v);

        let ray_direction = pixel_sample - self.center;

        Ray {
            origin: self.center,
            direction: ray_direction,
        }
    }

    pub fn render_to_file(&self, world: &dyn Hittable, path: &str) -> Result<()> {
        let mut buffer = format!("P3\n{} {}\n255\n", self.image_width, self.image_height);

        self.render(world, |_i, _j, pixel_color| {
            pixel_color
                .write_color(&mut buffer)
                .expect("Failed to write pixel color to buffer");
        });

        fs::write(path, buffer)?;
        println!("Done");

        Ok(())
    }
}

fn sample_square() -> Vec3 {
    vec3(
        rand::random::<f32>() - 0.5,
        rand::random::<f32>() - 0.5,
        0.0,
    )
}

#[derive(Clone, Copy)]
pub struct HitRecord {
    point: Vec3,
    normal: Vec3,
    t: f32,
    front_face: bool,
}

impl HitRecord {
    pub fn new(point: Vec3, t: f32, ray: &Ray, outward_normal: Vec3) -> Self {
        let front_face = ray.direction.dot(outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        };

        Self {
            point,
            t,
            normal,
            front_face,
        }
    }
}

pub trait Hittable {
    fn hit(&self, ray: &Ray, ray_t: Range<f32>) -> Option<HitRecord>;
}

struct Sphere {
    center: Vec3,
    radius: f32,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, ray_t: Range<f32>) -> Option<HitRecord> {
        let oc = self.center - ray.origin;
        let a = ray.direction.length_squared();
        let h = ray.direction.dot(oc);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let squared_discriminant = discriminant.sqrt();

        let mut root = (h - squared_discriminant) / a;
        if !ray_t.surrounds(root) {
            root = (h + squared_discriminant) / a;
            if !ray_t.surrounds(root) {
                return None;
            }
        }

        let point = ray.at(root);
        let outward_normal = (point - self.center) / self.radius;
        let hit_record = HitRecord::new(point, root, ray, outward_normal);

        Some(hit_record)
    }
}

impl Hittable for Vec<Box<dyn Hittable>> {
    fn hit(&self, ray: &Ray, ray_t: Range<f32>) -> Option<HitRecord> {
        let mut result = None;
        let mut closest_so_far = ray_t.end;

        for object in self {
            if let Some(hit_record) = object.hit(ray, ray_t.start..closest_so_far) {
                result = Some(hit_record);
                closest_so_far = hit_record.t;
            }
        }

        result
    }
}

trait RangeExt<Idx> {
    fn surrounds(&self, x: Idx) -> bool;

    fn clamp(&self, x: Idx) -> Idx;
}

impl<Idx> RangeExt<Idx> for Range<Idx>
where
    Idx: PartialOrd + Copy,
{
    fn surrounds(&self, x: Idx) -> bool {
        self.start < x && x < self.end
    }

    fn clamp(&self, x: Idx) -> Idx {
        if x < self.start {
            self.start
        } else if x > self.end {
            self.end
        } else {
            x
        }
    }
}

const EMPTY: Range<f32> = f32::INFINITY..-f32::INFINITY;
const UNIVERSE: Range<f32> = -f32::INFINITY..f32::INFINITY;

fn main() {
    let camera = Camera::new(400, 16.0 / 9.0);
    let world: Vec<Box<dyn Hittable>> = vec![
        Box::new(Sphere {
            center: vec3(0.0, 0.0, -1.0),
            radius: 0.5,
        }),
        Box::new(Sphere {
            center: vec3(0.0, -100.5, -1.0),
            radius: 100.0,
        }),
    ];

    camera
        .render_to_file(&world, "out/image.ppm")
        .expect("Failed to write image to file");
}
