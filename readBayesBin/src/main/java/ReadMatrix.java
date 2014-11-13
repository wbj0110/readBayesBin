package main.java;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.naivebayes.AbstractNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.classifier.naivebayes.ComplementaryNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.vectorizer.TFIDF;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;

public class ReadMatrix {
	public static Map<String, Integer> readDictionnary(Configuration conf, Path dictionnaryPath) {
		Map<String, Integer> dictionnary = new HashMap<String, Integer>();
		for (Pair<Text, IntWritable> pair : new SequenceFileIterable<Text, IntWritable>(dictionnaryPath, true, conf)) {
			dictionnary.put(pair.getFirst().toString(), pair.getSecond().get());
		}
		return dictionnary;
	}

	public static Map<Integer, Long> readDocumentFrequency(Configuration conf, Path documentFrequencyPath) {
		Map<Integer, Long> documentFrequency = new HashMap<Integer, Long>();
		for (Pair<IntWritable, LongWritable> pair : new SequenceFileIterable<IntWritable, LongWritable>(documentFrequencyPath, true, conf)) {
			documentFrequency.put(pair.getFirst().get(), pair.getSecond().get());
		}
		return documentFrequency;
	}
	

		public static void main(String[] args) {
			try {
				String modelPath = "D:\\work\\seq_bayes_get\\model\\";
				String labelIndexPath = "D:\\work\\seq_bayes_get\\labelindex";
				String dictionaryPath = "D:\\work\\seq_bayes_get\\dictionary.file-0";
				//String documentFrequencyPath = "D:\\work\\seq_bayes_get\\df-count\\df-count-merge";
				String documentFrequencyPath = "D:\\work\\seq_bayes_get\\df-count\\part-r-00001";
				String tweetsPath = "D:\\work\\seq_bayes_get\\test_txt\\54259";
				
				//bayesBin();
				Configuration configuration = new Configuration();
				NaiveBayesModel model = NaiveBayesModel.materialize(new Path(modelPath), configuration);
				StandardNaiveBayesClassifier classifier = new StandardNaiveBayesClassifier(model);


				// labels is a map label => classId
				Map<Integer, String> labels = BayesUtils.readLabelIndex(configuration, new Path(labelIndexPath));
				Map<String, Integer> dictionary = readDictionnary(configuration, new Path(dictionaryPath));
				Map<Integer, Long> documentFrequency = readDocumentFrequency(configuration, new Path(documentFrequencyPath));

				
				// analyzer used to extract word from tweet
				Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_43);
				
				int labelCount = labels.size();
				int documentCount = documentFrequency.get(-1).intValue();
				
				System.out.println("Number of labels: " + labelCount);
				System.out.println("Number of documents in training set: " + documentCount);
				BufferedReader reader = new BufferedReader(new FileReader(tweetsPath));
				String tweet ="";
				while(true) {
					String line = reader.readLine();
					if (line == null) {
						break;
					}
					tweet += line;
				}

				 Multiset<String> words =	ConcurrentHashMultiset.create();
					// extract words from tweet
					TokenStream ts = analyzer.tokenStream("text", new StringReader(tweet));
					CharTermAttribute termAtt = ts.addAttribute(CharTermAttribute.class);
					ts.reset();
					int wordCount = 0;
					while (ts.incrementToken()) {
						if (termAtt.length() > 0) {
							String word = ts.getAttribute(CharTermAttribute.class).toString();
							Integer wordId = dictionary.get(word);
							// if the word is not in the dictionary, skip it
							if (wordId != null) {
								words.add(word);
								wordCount++;
							}
						}
					}
					reader.close();
					// create vector wordId => weight using tfidf
					Vector vector = new RandomAccessSparseVector(10000);
					TFIDF tfidf = new TFIDF();
					for (Multiset.Entry<String> entry:words.entrySet()) {
						String word = entry.getElement();
						int count = entry.getCount();
						Integer wordId = dictionary.get(word);
						Long freq = documentFrequency.get(wordId);
						if(freq ==null)freq=1L;
						double tfIdfValue = tfidf.calculate(count, freq.intValue(), wordCount, documentCount);
						vector.setQuick(wordId, tfIdfValue);
					}
					// With the classifier, we get one score for each label 
					// The label with the highest score is the one the tweet is more likely to
					// be associated to
					Vector resultVector = classifier.classifyFull(vector);
					double bestScore = -Double.MAX_VALUE;
					int bestCategoryId = -1;
					for(Element element: resultVector.all()) {
						int categoryId = element.index();
						double score = element.get();
						if (score > bestScore) {
							bestScore = score;
							bestCategoryId = categoryId;
						}
						System.out.print("  " + labels.get(categoryId) + ": " + score);
					}
					System.out.println();
					System.out.println(" => " + labels.get(bestCategoryId));
				
				analyzer.close();
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		
		public static void bayesBin() throws Exception{
			Configuration configuration = new Configuration();

			// model is a matrix (wordId, labelId) => probability score
			NaiveBayesModel model = NaiveBayesModel.materialize(new Path("D:\\work\\seq_bayes_get\\model\\"), configuration);
			
			//NaiveBayesModel model = NaiveBayesModelUtil.materialize("D:\\work\\seq_bayes_get\\model\\naiveBayesModel.bin");
			System.out.println(model.numLabels());
		}
		
		public static void classFierBayes(Boolean complementary,NaiveBayesModel model){
			AbstractNaiveBayesClassifier classifier;
		      if (complementary) {
		        classifier = new ComplementaryNaiveBayesClassifier(model);
		      } else {
		        classifier = new StandardNaiveBayesClassifier(model);
		      }
		}
}
