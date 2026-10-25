# EcoSort AI

EcoSort AI is a mobile-friendly waste-sorting assistant that helps users
classify an image into one of three supported categories:

-   **Paper & Cardboard**
-   **Plastic Packaging**
-   **Organic Fruit & Vegetable Waste**

The app is designed to make waste-sorting guidance easier to access from
a phone. It supports image selection from the gallery and camera
capture, and provides an English/Hindi interface.

## Features

-   Upload an image from the phone gallery
-   Capture a photo with the device camera (where supported by the
    browser/device)
-   Classify supported waste categories
-   Show an uncertain result when the available evidence is insufficient
-   English and Hindi UI
-   Waste-sorting guidance
-   Sample images for quick checks
-   History/impact section, if enabled in the current build

## Model and limitations

**Document the exact model used by this repository before publishing.**
Include its name, source, license, inference location (on-device or
remote), and any API or network requirements.

A general-purpose image classifier is not the same as a waste-trained
classifier. It may recognize an object's shape without reliably
identifying its material (for example, distinguishing a plastic bottle
from a glass bottle). Predictions can also be wrong, so users should
verify uncertain results and follow local recycling rules.

Do not describe the app as 100% accurate. Only report model scores, test
results, or performance measurements that were actually measured with
the current code and real images.

## Getting started

The exact setup commands depend on the framework and package scripts in
this project.

1.  Clone or download this repository.
2.  Open the project folder.
3.  Install dependencies using the package manager and command
    documented in `package.json`.
4.  Start the development server using the project's actual start
    script.
5.  Open the local development URL shown in the terminal.

Before submitting, check `package.json` and replace this section with
the verified commands for this project. Do not copy example commands
unless the project supports them.

## How to use

1.  Open EcoSort AI.
2.  Choose an image from the gallery or take a photo using the camera
    option.
3.  Review the predicted category and disposal guidance.
4.  If the result is uncertain or the material cannot be identified
    reliably, take a clearer photo or follow local waste-management
    guidance.

## Testing checklist

Before release, test the current app with real images of:

-   A plastic bottle or container
-   Paper and a cardboard box
-   Fruit or vegetable scraps, such as a banana peel
-   Leaves
-   A person
-   An unrelated household object

Confirm that the app does not force unrelated images into a waste
category, and that it does not show misleading confidence for incorrect
results. Record only tests actually performed on the current build.

## Tech stack

Update this list to match the actual current source code:

-   Frontend framework: **\[verify from package.json\]**
-   Classification model: **\[add exact model name and source\]**
-   Inference: **\[on-device or remote; verify\]**
-   Language support: English and Hindi

## Privacy

Describe image handling accurately after checking the implementation.
State whether images stay on the device or are sent to a server/API, and
whether any analytics or storage are used. Do not claim that images
never leave the device unless the code confirms it.

## Known limitations

-   Classification depends on image quality, lighting, framing, and the
    model's training data.
-   Visual appearance alone may not reveal the material an item is made
    from.
-   Local recycling rules and accepted materials vary by location.
-   EcoSort AI is an aid, not a guarantee of correct recycling
    decisions.

## Contributing

Issues and improvements are welcome. Please include clear reproduction
steps and sample images when reporting a classification problem, while
avoiding photos that reveal private or sensitive information.

## License

Add a license only after confirming that you have the right to license
this project's source code and that its dependencies/model assets have
compatible terms. The model's license does not automatically determine
the license for the entire app.

## Project status

This README should be updated to match the exact source code and model
included in the public repository. Remove all bracketed placeholders and
verify setup instructions before submission.
