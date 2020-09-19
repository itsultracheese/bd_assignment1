import java.io.*;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.concurrent.TimeUnit;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Combined {

    public static class IDFMapper
            extends Mapper<Object, Text, Text, IntWritable>{

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            String line = new String(value.toString());
            String text = line.substring(line.indexOf(", \"text\":") + 9);
            StringTokenizer itr = new StringTokenizer(text.toLowerCase());
            String cur = "";
            HashSet used = new HashSet();

            while (itr.hasMoreTokens()) {
                cur = itr.nextToken().replaceAll("[^a-z\\-]", "");
                if(!used.contains(cur)){
                    used.add(cur);
                    word.set(cur);
                    context.write(word, one);
                }
            }
        }
    }

    public static class IDFReducer
            extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static class IndMapper
            extends Mapper<Object, Text, Text, MapWritable>{

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            String line = value.toString(); // acquiring the doc
            String text = line.substring(line.indexOf(", \"text\":") + 9); // getting doc text
            String id = line.substring(8, line.indexOf("\", \"url\"")); // getting doc id

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

        ///////////// IDF /////////////
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(Combined.class);
        job.setMapperClass(IDFMapper.class);
        job.setCombinerClass(IDFReducer.class);
        job.setReducerClass(IDFReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        Path path = new Path("output_idf");
        FileOutputFormat.setOutputPath(job, path);
        job.waitForCompletion(true);
        ///////////// IDF /////////////


        ///////////// Word2Vec & TF-IDF /////////////
        // setting configs
        conf = new Configuration();

        // reading IDF from file
        path = new Path("output_idf/part-r-00000");
        FileSystem fs = FileSystem.get(conf);
        BufferedReader reader;
        try {
            FSDataInputStream is = fs.open(path);
//            Path p = new Path("/user/vagrant/output_idf/part-r-00000");
            reader = new BufferedReader(new InputStreamReader(fs.open(path)));
            
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
        Job job2 = Job.getInstance(conf, "Indexer");

        // setting corresponding classes
        job2.setJarByClass(Combined.class);
        job2.setMapperClass(IndMapper.class);
        job2.setReducerClass(IndReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(MapWritable.class);

        // Files
        FileInputFormat.setInputDirRecursive(job2, true);
        FileInputFormat.addInputPath(job2, new Path(args[0]));
        FileOutputFormat.setOutputPath(job2, new Path(args[1]));

        // Starting the job
        System.exit(job2.waitForCompletion(true) ? 0 : 1);

        ///////////// Word2Vec & TF-IDF /////////////
    }
}
