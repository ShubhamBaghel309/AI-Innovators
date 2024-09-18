import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

# Load the original model
model = tf.keras.models.load_model('model2.h5')

# Define pruning parameters
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=pruning_steps)
}

# Define a pruning model
pruning_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile the pruning model
pruning_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prune the model
pruning_model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Save the pruned model
pruning_model.save('pruned_model.h5')
