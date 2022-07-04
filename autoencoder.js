function normalize(tensor, min, max) {
    const result = tf.tidy(function() {
        const MIN_VALUES = tf.scalar(min);
        const MAX_VALUES = tf.scalar(max);
    
        // Now calculate subtract the MIN_VALUE from every value in the Tensor
        // And store the results in a new Tensor.
        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    
        // Calculate the range size of possible values.
        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    
        // Calculate the adjusted values divided by the range size as a new Tensor.
        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
        
        // Return the important tensors.
        return NORMALIZED_VALUES;
      });
      return result;
}

class DenoinsingAutoencoder {
    constructor(input_data, output_data) {
        this.input_data = normalize(tf.tensor(input_data), 0, 255);
        this.output_data = normalize(tf.tensor(output_data), 0, 255);

        this.build_model();
        this.full_model.compile({
            optimizer: 'adam', // Adam changes the learning rate over time which is useful.
            loss: 'meanSquaredError',
            metrics: ['accuracy']
        });
    }

    build_model() {
        let h = this.input_data.shape[1];
        let w = this.input_data.shape[2];
        let channels = this.input_data.shape[3];
        
        this.encoder = this.build_encoder(h, w, channels);
        this.encoder.summary();

        this.decoder = this.build_decoder(h, w, channels);
        this.decoder.summary();

        this.full_model = tf.sequential();
        this.full_model.add(this.encoder);
        this.full_model.add(this.decoder);
        this.full_model.summary();
    }

    build_encoder(h, w, channels) {
        let encoder = tf.sequential();
        encoder.add(tf.layers.upSampling2d ({
            inputShape: [w, h, channels], 
            size: [2, 2]
        }));
        encoder.add(tf.layers.conv2dTranspose ({
            filters: 3, 
            kernelSize: 5
        }));

        encoder.add(tf.layers.conv2dTranspose ({
            filters: 4, 
            kernelSize: 5
        }));
        encoder.add(tf.layers.conv2dTranspose ({
            filters: 8, 
            kernelSize: 5
        }));
        return encoder;
    }

    build_decoder(h, w, channels) {
        let sample_tensor = tf.zeros([1, h, w, channels]);
        let latent_shape = this.encoder.predict(sample_tensor).shape.slice(1);

        let decoder = tf.sequential();
        decoder.add(tf.layers.conv2d({
            inputShape: latent_shape, 
            filters: 8, 
            kernelSize: 5
        }));
        decoder.add(tf.layers.maxPooling2d({
            poolSize: 2
        }));
        decoder.add(tf.layers.conv2d({
            filters: 3, 
            kernelSize: 5
        }));
        return decoder;
    }

    async train (epochs, log_fn){
        let results = await this.full_model.fit(this.input_data, this.output_data, {
            shuffle: true,        // Ensure data is shuffled again before using each time.
            validationSplit: 0.1,
            batchSize: 5,       // Update weights after every 500 examples.      
            epochs: epochs,         
            callbacks: {onBatchEnd: log_fn}
        });
        return results;
    }
}