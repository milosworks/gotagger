# GoTagger

An image tagger base using the `onnxruntime`. Because of extensibility, you will also need to interact with `onnxruntime_go`.

A good and recommended model to use is [SmilingWolf/wd-vit-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3/tree/v1.0), this is an example code:

```go
package main

import (
	"cmp"
	"fmt"
	"image"
	_ "image/png"
	"maps"
	"os"
	"slices"
	"strings"

	"github.com/milosworks/gotagger"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	modelPath = "./model.onnx"
	tagsPath  = "./selected_tags.csv"
	imagePath = "./image.png"
	// Example for windows
	// Runtimes are not provided since onnxruntime_go *sometimes* handles it
	// it is still recommended to specify the runtime as onnxruntime_go could have outdated runtimes
	runtimePath = "./onnxruntime.dll"
)

func main() {
	// You need to specify the shared library path to ort specifically, not gotagger
	ort.SetSharedLibraryPath(runtimePath)
	// VERY IMPORTANT: Initialize ort environment
	err := ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	// Don't forget to destroy it at the end
	defer ort.DestroyEnvironment()

	// Create a new gotagger session
	session, err := gotagger.New(modelPath, tagsPath)
	if err != nil {
		panic(err)
	}

	// Open example file
	file, err := os.Open(imagePath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Decode it into the image.Image
	img, _, err := image.Decode(file)
	if err != nil {
		panic(err)
	}

	// Run the session with the provided images
	// The library is smart enough to do images in batches even when the model has a batchsize of 1.
	// You can provide the GeneralThreshold and CharacterThreshold for their respective tags.
	// Also enable mcut on general or character tags
	predictions, err := session.Run(
		[]image.Image{img},
		gotagger.DefaultGeneralThreshold,
		gotagger.DefaultCharacterThreshold,
		false,
		false,
	)
	if err != nil {
		panic(err)
	}

	println("Predictions:")
	for i, pred := range predictions {
		fmt.Printf("Image [%d]: %s\n", i+1, strings.Join(pred.Names(), ", "))
		if len(pred.Character) != 0 {
			chars := slices.Collect(maps.Keys(pred.Character))
			slices.SortFunc(chars, func(a, b string) int {
				return cmp.Compare(pred.Character[b], pred.Character[a])
			})

			fmt.Printf("Character: %v\n", chars[0])
		}

		ratings := slices.Collect(maps.Keys(pred.Rating))
		slices.SortFunc(ratings, func(a, b string) int {
			return cmp.Compare(pred.Rating[b], pred.Rating[a])
		})

		fmt.Printf("Rating: %v\n", ratings[0])
	}
}

```

Input image
![gilgamesh.png](https://i.imgur.com/SfF7GRR.png)

Output predictions:
```
Predictions:
Image [1]: 1boy, blonde hair, male focus, red eyes, earrings, jewelry, short hair, gauntlets, solo, single gauntlet, turban, open mouth, tattoo, looking at viewer, sitting, throne, horns, arabian clothes, hair between eyes, own hands together, upper body
Character: gilgamesh (caster) (fate)
Rating: sensitive
```