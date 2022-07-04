import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';

let inputs;
let outputs;

const input_canvas = document.getElementById('input_canvas');
const output_canvas = document.getElementById('output_canvas');

let idx = 0;
let idx_span;
let idx_slider;
let button;
let train = false;
let max_noise = 0;
let max_noise_span;
let max_noise_slider;

let model;

function log_fn(batch, logs) {
    let img = outputs[idx];
    let noised_img = apply_noise(img, max_noise);
    drawImage(noised_img, input_canvas, 4);
    let result_batch = tf.tidy(() => {
        let normalized_batch = normalize(tf.tensor([noised_img]), 0, 255);

        let result_batch = model.full_model.predict(normalized_batch);
        return result_batch.mul(tf.scalar(255));
    });
    let result_img = result_batch.arraySync()[0];
    drawImage(result_img, output_canvas, 4);
    result_batch.dispose();
}

function create_dataset(input_data, n){
    // https://stackoverflow.com/questions/22464605/convert-a-1d-array-to-2d-array
    let new_inputs = [];
    let new_outputs = [];
    let k = int(input_data['length'] / n);
    for (let i = 0; i < input_data['length']; i = i + k) {
        let arr = input_data[i];
        let newArr = [];
        let n = sqrt(arr.length);
        while(arr.length){
            let line = arr.splice(0,n);
            let new_line = [];
            for (const pixel of line) {
                new_line.push([pixel*255, pixel*255, pixel*255]);
            }
            newArr.push(new_line);
        }

        let noise = random(128, 255);
        new_inputs.push(apply_noise(newArr, noise));
        new_outputs.push(newArr);
    }
    return {inputs: new_inputs, outputs: new_outputs};


}

function setup() {
    let dataset = create_dataset(TRAINING_DATA.inputs, 50);
    inputs = dataset['inputs'];
    outputs = dataset['outputs'];

    noCanvas();
    idx_span = createSpan('Image selector');
    idx_slider = createSlider(0, outputs.length-1, 100);
    max_noise_span = createSpan('Noise');
    max_noise_slider = createSlider(0, 255, 100);

    drawImage(outputs[idx], input_canvas, 4);

    button = createButton('train');
    button.mousePressed(async () => {
        if(!train) {
            train = true;
            button.html('stop');
            while (train) {
                await model.train(1, log_fn);
            }
        }
        else {
            button.html('train');
            train = false;  
        }
    });

    model = new DenoinsingAutoencoder(inputs, outputs);
}

async function draw() {
    if (max_noise != max_noise_slider.value() || idx != idx_slider.value()){
        max_noise = max_noise_slider.value();
        idx = idx_slider.value()
        let img = outputs[idx];
        let noised_img = apply_noise(img, max_noise);
        drawImage(noised_img, input_canvas, 4);

        let result_img = tf.tidy(() => {
            let normalized_batch = normalize(tf.tensor([noised_img]), 0, 255);

            let result_batch = model.full_model.predict(normalized_batch);
            return result_batch.mul(tf.scalar(255)).arraySync()[0];
        });
        drawImage(result_img, output_canvas, 4);
    }
    
    idx_span.position(10, input_canvas.height + 15);
    idx_slider.position(10, input_canvas.height + 30);
    
    max_noise_span.position(10, input_canvas.height + 45);
    max_noise_slider.position(10, input_canvas.height + 60);
    
    button.position(10, input_canvas.height + 80);
}

function apply_noise(img, max_noise) {
    let h = img.length;
    let w = img[0].length;
    let channels = img[0][0].length;
    let new_img = [];
    for(let y = 0; y < h; y++) {
        let line = [];
        for (let x = 0; x < w; x++) {
            let pixel = [];
            for (let c = 0; c < channels; c++) {
                let noise = constrain(img[y][x][c] + random(-max_noise, max_noise), 0 , 255);
                pixel.push(noise);
            }
            line.push(pixel);
        }
        new_img.push(line);
    }
    return new_img;
}

function drawImage(img, canvas, scale) {
    let h = img.length;
    let w = img[0].length;
    let channels = img[0][0].length;

    let ctx = canvas.getContext('2d');
    ctx.canvas.height = h * scale;
    ctx.canvas.width = w * scale;
    
    let imageData = ctx.getImageData(0, 0, h * scale, w * scale);

    for(let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            for (let y_offset = 0; y_offset < scale; y_offset++) {
                for (let x_offset = 0; x_offset < scale; x_offset++) {
                    let pixel_idx = (y * w * scale*scale * 4) + (y_offset * w *scale * 4) + (x * scale * 4) + (x_offset * 4);
                    imageData.data[pixel_idx] = img[y][x][0];      // Red Channel.
                    imageData.data[pixel_idx + 1] = img[y][x][1];  // Green Channel.
                    imageData.data[pixel_idx + 2] = img[y][x][2];  // Blue Channel.
                    imageData.data[pixel_idx + 3] = 255;        // Alpha Channel.
                }
            }
        }
    }

    // Render the updated array of data to the canvas itself.
    ctx.putImageData(imageData, 0, 0); 
}

window.setup = setup;
window.draw  = draw;