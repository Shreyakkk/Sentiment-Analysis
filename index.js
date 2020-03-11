var LSTM_URLS = {
	// model: 'https://language.googleapis.com/$discovery/rest?version=v1',
	// metadata: 'https://language.googleapis.com/$discovery/rest?version=v1beta2'
  model:
      'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
  metadata:
      'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
};
function status(m) {
  document.getElementById('status').innerHTML='';
  document.getElementById('status').innerHTML=m;
}
async function url_live(url) {
  // Test if an url can be accessed.
  status('Testing url ' + url, '20');
  try {
    var response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}
class SentimentAnalyser {
  // Main class for doing Sentiment analysis.
  async init(urls) {
    // Load both pre-trained model and its metadata.
    // urls - an object/dictonary with model/metadata
    //        keys that points out to URLS for hosted resources.
    this.urls = urls;
    status('Loading model from:' + urls.model);
    this.model = await this.load_model(urls.model);//Loading the model
    status('Loading metadata from:' + urls.metadata);
    await this.load_meta();//Loading meta data
    return this;
  }
  async load_model(url) {
    // Load a pre-trained model from a specific URL.
    status('Loading pretrained model from ' + url);
    try {
      var model = await tf.loadLayersModel(url);
      status('Done loading pretrained model.', '80');
      return model;
    } catch (err) {
      console.error(err);
      status('Loading pretrained model failed.');
    }
  }
  async get_meta(url) {
    // Get the metadata for our model
    // and turn it into an object that
    // we can access.
    status('Loading metadata from ' + url);
    try {
      var metadataJson = await fetch(url);
      var metadata = await metadataJson.json();
      status('Done loading metadata.', '100', true);
      return metadata;
    } catch (err) {
      console.error(err);
      status('Loading metadata failed.');
    }
  }
  async load_meta() {
    // Get metadata and copy some important
    // values as an attributes to our class
    // so we can later easily use them
    // in predict when we will be doing analysis.
    var sentimentMetadata = await this.get_meta(this.urls.metadata);
    this.indexFrom = sentimentMetadata['index_from'];
    this.maxLen = sentimentMetadata['max_len'];
    console.log('indexFrom = ' + this.indexFrom);
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = sentimentMetadata['word_index'];
  }
  predict(text) {
// Convert to lower case and remove all punctuations.
    const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' '); // Convert the words to a sequence of word indices.
    const sequence = inputText.map(word => {
      let wordIndex = this.wordIndex[word] + this.indexFrom;

      if (wordIndex > this.vocabularySize) {
        wordIndex = 2;
      }
      return wordIndex;
    }); // Perform truncation and padding.
    const paddedSequence = (0, padSequences)([sequence], this.maxLen);
    const input = tf.tensor2d(paddedSequence, [1, this.maxLen]);
    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    const score = predictOut.dataSync()[0];
    predictOut.dispose();
    const endMs = performance.now();
    console.log(score);
    return score;
  }
};
async function setup_analyser(urls) {
  // Just create a new analyser with specific
  // available model/metadata URL.
  if (await url_live(urls.model)) {
    var analyser = await new SentimentAnalyser().init(urls);
    return analyser;
  }
}
// Make a new analyser available globally.
// Usually not a good idea in production code.
// Always put it into a separate name space if
// in production if you can.
// To use LSTM model just change CNN_URLS to LSTM_URLS.
var analyser=setup_analyser(LSTM_URLS);// LSTM url contains both mopdel as well as meta data
let ready_analyser=null;

function calc_and_show(a) {
 var data=document.getElementById('review').value;
 var score=a.predict(data);
 console.log("score"+score);
 var pscore=score.toFixed(3);
 console.log("pscore = "+pscore);
 if (score == 0.654) {
 	m="neutral";
 }
 else if (score < 0.654) {
   m='negative';
 }
 else  {
 	m='positive';
 }
 console.log(data);
 document.getElementById('results').innerHTML="score = "+pscore+m;
 return a;
}

function padSequences(
    sequences, maxLen, padding = 'pre', truncating = 'pre', value = 0) {
  // TODO(cais): This perhaps should be refined and moved into tfjs-preproc.
  return sequences.map(seq => {
    // Perform truncation.
    if (seq.length > maxLen) {
      if (truncating === 'pre') {
        seq.splice(0, seq.length - maxLen);
      } else {
        seq.splice(maxLen, seq.length - maxLen);
      }
    }

    // Perform padding.
    if (seq.length < maxLen) {
      const pad = [];
      for (let i = 0; i < maxLen - seq.length; ++i) {
        pad.push(value);
      }
      if (padding === 'pre') {
        seq = pad.concat(seq);
      } else {
        seq = seq.concat(pad);
      }
    }
    return seq;
  });
}
function init() {
 	var data='';
 data = data.replace(/(\r\n|\n|\r)/gm, "");
 document.getElementById('review').value='';
 document.getElementById('results').innerHTML='';
 document.getElementById('review').value=data;
 document.getElementById('results').innerHTML='Calculating...'

 analyser.then(function(a) { ready_analyser=a;calc_and_show(a);status('');});

 let r = document.getElementById('review');
 r.addEventListener('input', function(e) {
     document.getElementById('results').innerHTML='Calculating...'
     calc_and_show(ready_analyser);
 });
}

init();