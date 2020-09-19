import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.join.TupleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Indexer {

    public static class IndMapper
            extends Mapper<Object, Text, Text, MapWritable>{

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            
            String line = value.toString(); // acquiring the doc
            String text = line.substring(line.indexOf(", \"text\":") + 9); // getting doc text
            String id = line.substring(line.indexOf(8, line.indexOf("\", \"url\""))); // getting doc id

            StringTokenizer itr = new StringTokenizer(text.toLowerCase()); // iterating through text
            String cur = ""; // current word in text
            
            MapWritable map = new MapWritable(); // <word, # of its occurences in the text>
            
            // iterating through text
            while (itr.hasMoreTokens()) {
                cur = itr.nextToken().replaceAll("[^a-z\\-]", ""); //cur word
                if (! cur.equals("")) {
                    word = new Text(cur);
                    if (! map.containsKey(word)) {
                        map.put(word, one);
                    }
                    else {
                        IntWritable r = (IntWritable) map.get(word);
                        map.put(word, new IntWritable(r.get() + 1));
                    }
                }
            }

            context.write(new Text(id), map); // passing to reducer
        }
    }

    public static class IndReducer
            extends Reducer<Text, MapWritable, Text, MapWritable> {

        public void reduce(Text key, Iterable<MapWritable> values,
                           Context context
        ) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration(); // get config
            MapWritable result = new MapWritable(); // <hash of word, tf-idf of word>

            for (MapWritable map: values) {
                Set<Writable> keys = map.keySet(); // getting all words that occurred in text
                for(Writable k: keys){
                    String word = ((Text) k).toString();
                    Integer tf = ((IntWritable)(map.get(k))).get();
                    Integer idf = conf.getInt(word, -1);
                    Float tfidf = (float)tf / (float)idf;

                    result.put(new IntWritable(word.hashCode()), new FloatWritable(tfidf));
                }
            }

            context.write(key, result); // writing the result
        }
    }

    public static void main(String[] args) throws Exception {
        // setting configs
        Configuration conf = new Configuration();

        // reading IDF from file
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader("output/part-r-00000"));
            String line = reader.readLine();
            while(line != null){
                StringTokenizer itr = new StringTokenizer(line);
                String cur_word = "";
                Integer cur_idf = 0;
                // iterating through line
                if(itr.hasMoreTokens()){
                    cur_word = itr.nextToken();
                    if (itr.hasMoreTokens()) {
                        cur_idf = Integer.parseInt(itr.nextToken().replaceAll("[^0-9]", ""));
                        conf.setInt(cur_word, cur_idf); //passing <word, idf> to mapreduce
                    }
                }

                line = reader.readLine(); // reading the next line
            }
        } catch (IOException e){
            e.printStackTrace();
        }

        // initializing job
        Job job = Job.getInstance(conf, "Indexer");

        // setting corresponding classes
        job.setJarByClass(Indexer.class);
        job.setMapperClass(IndMapper.class);
        job.setReducerClass(IndReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // Files
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // Starting the job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
