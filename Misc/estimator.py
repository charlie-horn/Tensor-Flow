from Misc.get_data import *

deep_model = estimator.DNNClassifier(hidden_units=[13,13,13],
                            feature_columns=feat_cols,
                            n_classes=3,
                            optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01) )

input_fn = estimator.inputs.numpy_input_fn(x={'x':scaled_x_train},y=y_train,shuffle=True,batch_size=10,num_epochs=5)

deep_model.train(input_fn=input_fn,steps=500)

input_fn_eval = estimator.inputs.numpy_input_fn(x={'x':scaled_x_test},shuffle=False)

preds = list(deep_model.predict(input_fn=input_fn_eval))

predictions = [p['class_ids'][0] for p in preds]

print(classification_report(y_test,predictions))