package gotagger

import (
	"cmp"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"maps"
	"math"
	"os"
	"slices"
	"strings"

	"github.com/disintegration/imaging"

	"github.com/go-gota/gota/dataframe"
	ort "github.com/yalue/onnxruntime_go"
)

var kaomojis = map[string]struct{}{
	"0_0":     {},
	"(o)_(o)": {},
	"+_+":     {},
	"+_-":     {},
	"._.":     {},
	"<o>_<o>": {},
	"<|>_<|>": {},
	"=_=":     {},
	">_<":     {},
	"3_3":     {},
	"6_9":     {},
	">_o":     {},
	"@_@":     {},
	"^_^":     {},
	"o_o":     {},
	"u_u":     {},
	"x_x":     {},
	"|_|":     {},
	"||_||":   {},
}

const (
	// DefaultGeneralThreshold is the default threshold for all general tags
	DefaultGeneralThreshold float32 = 0.35
	// DefaultCharacterThreshold is the default threshold for all character tags
	DefaultCharacterThreshold float32 = 0.85
)

type modelTags struct {
	names            []string
	ratingIndexes    []int
	generalIndexes   []int
	characterIndexes []int
}

// TaggerSession is the representation of the ORT session for this tagger
type TaggerSession struct {
	modelTags
	input      ort.Shape
	output     ort.Shape
	targetSize int
	batchSize  int
	Session    *ort.DynamicSession[float32, float32]
}

func loadTags(tagsPath string) (modelTags, error) {
	csvFile, err := os.Open(tagsPath)
	if err != nil {
		return modelTags{}, fmt.Errorf("error while trying to open file %s: %w", tagsPath, err)
	}
	defer csvFile.Close()

	df := dataframe.ReadCSV(csvFile)
	nameCol := df.Col("name").Records()
	names := make([]string, len(nameCol))

	for i, record := range nameCol {
		if _, ok := kaomojis[record]; !ok {
			names[i] = strings.ReplaceAll(record, "_", " ")
		} else {
			names[i] = record
		}
	}

	categoryCol := df.Col("category").Records()

	var (
		ratingIndexes    []int
		generalIndexes   []int
		characterIndexes []int
	)

	for i, record := range categoryCol {
		switch record {
		case "9":
			ratingIndexes = append(ratingIndexes, i)
		case "0":
			generalIndexes = append(generalIndexes, i)
		case "4":
			characterIndexes = append(characterIndexes, i)
		}
	}

	return modelTags{names, ratingIndexes, generalIndexes, characterIndexes}, nil
}

// New creates a new TaggerSession with the provided model and tags dataset path.
//
// It is important to initialize and set the shared library for ORT before calling this function.
func New(modelPath string, tagsPath string) (TaggerSession, error) {
	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return TaggerSession{}, fmt.Errorf(
			"error while getting input/output info of modelPath %s: %w",
			modelPath,
			err,
		)
	}

	input := inputs[0]
	output := outputs[0]

	session, err := ort.NewDynamicSession[float32, float32](
		modelPath,
		[]string{input.Name},
		[]string{output.Name},
	)
	if err != nil {
		return TaggerSession{}, fmt.Errorf("error while starting new dynamic session: %w", err)
	}

	tags, err := loadTags(tagsPath)
	if err != nil {
		return TaggerSession{}, err
	}

	inputShape := input.Dimensions

	return TaggerSession{
		modelTags:  tags,
		input:      inputShape,
		output:     output.Dimensions,
		batchSize:  int(inputShape[0]),
		targetSize: int(inputShape[1]),
		Session:    session,
	}, nil
}

func prepareInput(img image.Image, targetSize int) []float32 {
	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()

	maxDim := w
	if h > maxDim {
		maxDim = h
	}
	padded := imaging.New(maxDim, maxDim, color.White)
	offset := image.Pt(
		(maxDim-bounds.Dx())/2,
		(maxDim-bounds.Dy())/2,
	)

	processedImg := imaging.Paste(padded, img, offset)

	if maxDim != targetSize {
		processedImg = imaging.Resize(processedImg, targetSize, targetSize, imaging.Lanczos)
	}

	data := make([]float32, 0, 3*targetSize*targetSize)

	for y := 0; y < targetSize; y++ {
		for x := 0; x < targetSize; x++ {
			r, g, b, _ := processedImg.At(x, y).RGBA()

			data = append(data, float32(b>>8), float32(g>>8), float32(r>>8))
		}
	}

	return data
}

// Predictions is the output of the Run function containing all tags
type Predictions struct {
	General   map[string]float32
	Rating    map[string]float32
	Character map[string]float32
}

// Names will output the sorted General tags names
func (p *Predictions) Names() []string {
	q := slices.Collect(maps.Keys(p.General))

	slices.SortFunc(q, func(a, b string) int {
		return cmp.Compare(p.General[b], p.General[a])
	})

	return q
}

func mcutThreshold(probs []float32) float32 {
	if len(probs) < 2 {
		if len(probs) == 0 {
			return 0
		}
		return probs[0]
	}

	sortedProbs := make([]float32, len(probs))
	copy(sortedProbs, probs)
	slices.SortFunc(sortedProbs, func(a, b float32) int {
		if a > b {
			return -1
		} else if a < b {
			return 1
		}
		return 0
	})
	maxDiff := float32(0)
	maxIndex := 0
	for i := 0; i < len(sortedProbs)-1; i++ {
		diff := sortedProbs[i] - sortedProbs[i+1]
		if diff > maxDiff {
			maxDiff = diff
			maxIndex = i
		}
	}
	return (sortedProbs[maxIndex] + sortedProbs[maxIndex+1]) / 2
}

// Run the current session with the provided images and settings
//
// An easy example would be:
//
//		session.Run(
//		 []image.Image{img},
//		 gotagger.DefaultGeneralThreshold,
//	  gotagger.DefaultCharacterThreshold,
//	  false, false
//	 )
//
// All tags that have a prediction higher to the threshold will fall into the output.
// You can use mcut threshold for the general and character tags, for more information check:
// https://search.r-project.org/CRAN/refmans/utiml/html/mcut_threshold.html
func (s *TaggerSession) Run(
	images []image.Image,
	generalThreshold float32,
	characterThreshold float32,
	generalMCutEnabled bool,
	characterMCutEnabled bool,
) ([]Predictions, error) {
	predictions := make([]Predictions, 0, len(images))
	size := int(math.Abs(float64(s.batchSize)))
	chunks := make([][]image.Image, 0, size)
	if s.batchSize == -1 {
		chunks = append(chunks, images)
	} else {
		chunks = slices.Collect(slices.Chunk(images, s.batchSize))
	}

	for _, chunk := range chunks {
		imgData := make([]float32, 0, size*3*(s.targetSize^2))

		for _, img := range chunk {
			imgData = append(imgData, prepareInput(img, s.targetSize)...)
		}

		inShape := s.input
		if inShape[0] == -1 {
			inShape[0] = int64(size)
		}

		inTensor, err := ort.NewTensor(inShape, imgData)
		if err != nil {
			return nil, fmt.Errorf("error ocurred when creating input tensor: %w", err)
		}

		outShape := s.output
		if outShape[0] == -1 {
			outShape[0] = int64(size)
		}

		outSize := int(outShape[1])

		outTensor, err := ort.NewEmptyTensor[float32](outShape)
		if err != nil {
			return nil, fmt.Errorf("error ocurred when creating output tensor: %w", err)
		}

		err = s.Session.Run([]*ort.Tensor[float32]{inTensor}, []*ort.Tensor[float32]{outTensor})
		if err != nil {
			return nil, fmt.Errorf("error ocurred when running session: %w", err)
		}

		out := outTensor.GetData()
		for i := 0; i < len(chunk); i++ {
			data := out[outSize*(i) : outSize*(i+1)]

			computedGeneralThreshold := generalThreshold
			if generalMCutEnabled {
				var generalProbs []float32
				for _, index := range s.generalIndexes {
					if index < len(data) {
						generalProbs = append(generalProbs, data[index])
					}
				}
				computedGeneralThreshold = mcutThreshold(generalProbs)
			}

			computedCharacterThreshold := characterThreshold
			if characterMCutEnabled {
				var characterProbs []float32
				for _, index := range s.characterIndexes {
					if index < len(data) {
						characterProbs = append(characterProbs, data[index])
					}
				}
				computedCharacterThreshold = mcutThreshold(characterProbs)
				if computedCharacterThreshold < 0.15 {
					computedCharacterThreshold = 0.15
				}
			}

			p := Predictions{
				General:   map[string]float32{},
				Rating:    map[string]float32{},
				Character: map[string]float32{},
			}
			for index, pred := range data {
				name := s.names[index]

				if slices.Contains(s.ratingIndexes, index) {
					p.Rating[name] = pred
				}
				if slices.Contains(s.generalIndexes, index) && pred > computedGeneralThreshold {
					p.General[name] = pred
				}
				if slices.Contains(s.characterIndexes, index) && pred > computedCharacterThreshold {
					p.Character[name] = pred
				}
			}

			predictions = append(predictions, p)
		}

		outTensor.Destroy()
		inTensor.Destroy()
	}

	return predictions, nil
}

// Destroy the current session
func (s *TaggerSession) Destroy() error {
	return s.Session.Destroy()
}
