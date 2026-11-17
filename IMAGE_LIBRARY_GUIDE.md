# Image Libraries

The desktop UI includes two reusable image galleries:

- **Face Library** beside **Select a face** stores source-face images.
- **Background Library** in the Camera panel stores replacement backgrounds for the 3D background effect.

Open either gallery and choose **Import images** to copy one or more supported images into the project. Select a thumbnail to use that image. The application creates these folders automatically:

- `image_library/faces/`
- `image_library/backgrounds/`

Supported formats follow the app's existing image extensions: PNG, JPG, JPEG, BMP, and WEBP. Imported images are copied into the library; the original files are not moved or modified. Keep only images you have permission to use, and obtain consent before using a real person's face.


## Included sample images

The `image_library` folder now includes six fictional, AI-generated visual test assets:

- `faces/Sample_Male_Face.jpg` — fictional adult male portrait
- `faces/Sample_Female_Face.jpg` — fictional adult female portrait
- `backgrounds/Sample_Office.jpg` — modern office interior
- `backgrounds/Sample_Living_Room.jpg` — contemporary living room
- `backgrounds/Sample_City_Night.jpg` — city skyline at night
- `backgrounds/Sample_Tropical_Beach.jpg` — tropical beach scene

These are synthetic demonstration images, not photographs of identified real people. They are included for local UI and image-processing tests. Face-swap testing should only use faces of fictional people or real people who have given informed consent. Clearly disclose and label synthetic or altered media when sharing it.

## Included room background pack

The app also ships with an additional 11-room image pack in `image_library/backgrounds/` for 3D background use. Each image has a matching depth preset stored in `image_library/backgrounds/background_pack.json` so that selecting a room automatically applies a recommended parallax, perspective, and blur setting.

This pack is intended for virtual room / studio / interior styles and is suited to the live 2.5D background pipeline already built into the app.
