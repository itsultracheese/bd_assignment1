import java.io.*;
import java.net.URI;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.concurrent.TimeUnit;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Indexer {

    public static class IDFMapper
            extends Mapper<Object, Text, Text, IntWritable>{

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            String line = value.toString(); // reading a ling containing json about doc
            String text = line.substring(line.indexOf(", \"text\":") + 9); // acquiring "text" part of json

            StringTokenizer itr = new StringTokenizer(text.toLowerCase()
                    .replaceAll("\\\\[a-z]", " ")
                    .replaceAll("-", " ")); //tokenizing text, removing symbols of tabulation and "-"

            line = null;
            text = null;

            String cur; // current token
            HashSet used = new HashSet(); // stores already processed tokens (words)

            // iterating through tokens
            while (itr.hasMoreTokens()) {
                // getting rid of non-letter symbols
                cur = itr.nextToken().replaceAll("[\\\\0-9~`!@#$%^&*()\\-_+=\\,.<>?/'\":;{}\\[\\]\\|]", "");
                if (! cur.equals("")) { // non-empty token
                    if (!used.contains(cur)) { // 1st occurrence
                        used.add(cur); // adding to processed tokens
                        word.set(cur);
                        context.write(word, one); // passing to reducer
                    }
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
            int sum = 0; // # of docs in which the word has occurred
            for (IntWritable val : values) {
                sum += val.get();
            }

            result.set(sum);
            context.write(key, result);
        }
    }

    public static class SORTIDFMapper
            extends Mapper<Object, Text, IntWritable, Text>{

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            StringTokenizer itr = new StringTokenizer(value.toString());
            String word;
            int idf;

            if (itr.hasMoreTokens()){
                word = itr.nextToken();
                if (itr.hasMoreTokens()) {
                    idf = Integer.parseInt(itr.nextToken().replaceAll("[^0-9]", ""));
                    context.write(new IntWritable(idf), new Text(word));
                }
            }
        }
    }

    public static class SORTIDFComparator
            extends WritableComparator {

        protected SORTIDFComparator() {
            super(IntWritable.class, true);
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            IntWritable f1 = (IntWritable) a;
            IntWritable f2 = (IntWritable) b;
            return f2.compareTo(f1);
        }
    }

    public static class SORTIDFReducer
            extends Reducer<IntWritable,Text,Text,IntWritable> {

        public void reduce(IntWritable key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            for(Text val: values) {
                context.write(val, key);
            }
        }
    }

    public static class IndMapper
            extends Mapper<Object, Text, Text, MapWritable>{

        private Text word;

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            String line = value.toString(); // acquiring the doc
            String text = line.substring(line.indexOf(", \"text\":") + 9); // getting doc text
            String id = line.substring(8, line.indexOf("\", \"url\"")); // getting doc id

            StringTokenizer itr = new StringTokenizer(text.toLowerCase().
                    replaceAll("\\\\[a-z]", " ")
                    .replaceAll("-", " ")); //tokenizing text, removing symbols of tabulation and "-"

            String cur; // current token


            text = null;
            line = null;

            MapWritable map = new MapWritable(); // <word, # of its occurrences in the text>

            // iterating through text
            while (itr.hasMoreTokens()) {
                // getting rid of non-letter symbols
                cur = itr.nextToken().replaceAll("[\\\\0-9~`!@#$%^&*()\\-_+=\\,.<>?/'\":;{}\\[\\]\\|]", "");
                if (! cur.equals("")) { //not empty tokens
                    word = new Text(cur);
                    if (! map.containsKey(word)) {
                        map.put(word, new IntWritable(1));
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

        private static HashMap<String, Integer> get_idf = new HashMap<String, Integer>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration(); // get config
            
            // reading files from distributed cache
            
            URI[] cacheFiles = context.getCacheFiles();

            for (URI cf: cacheFiles){
                FileSystem fs = FileSystem.get(conf);
                Path path = new Path(cf.toString());
                BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path)));

                String line = reader.readLine(); // reading the 1st line of the file
                while(line != null){
                    StringTokenizer itr = new StringTokenizer(line);
                    String cur_word;
                    Integer cur_idf;
                    // iterating through line
                    if(itr.hasMoreTokens()){
                        cur_word = itr.nextToken();
                        if (itr.hasMoreTokens()) {
                            cur_idf = Integer.parseInt(itr.nextToken().replaceAll("[^0-9]", ""));
                            get_idf.put(cur_word, cur_idf);
                        }
                    }

                    line = reader.readLine(); // reading the next line
                }

            }
        }

        public void reduce(Text key, Iterable<MapWritable> values,
                           Context context
        ) throws IOException, InterruptedException {

            MapWritable result = new MapWritable(); // <hash of word, tf-idf of word>

            Integer length = 0; // # of words in doc text
            for (MapWritable map: values) {
                Set<Writable> keys = map.keySet(); // getting all words that occurred in text
                for(Writable k: keys){
                    String word = k.toString();
                    Integer tf = ((IntWritable)(map.get(k))).get(); // tf of a word in the given doc

                    length = length + tf; // updating the length of the doc


                    Integer idf = get_idf.get(word);

                    if (idf != null) { //making sure that word occurred at least in 1 doc
                        Float tfidf = (float)tf / (float)idf;
                        result.put(new IntWritable(word.hashCode()), new FloatWritable(tfidf));
                    }

                }
            }

            context.write(new Text(key.toString() + " length: " + length.toString()), result); // writing the result
        }
    }

    public static void args_usage() {
        System.out.println("Arguments usage:");
        System.out.println("hadoop jar Indexer.java Indexer arg0 arg1");
        System.out.println("arg0 - path to folder with wiki dump");
        System.out.println("arg1 - path to output folder (should not exist before execution");
        System.out.println("---------------------------------");
        System.out.println("Example: hadoop jar Indexer.java Indexer /EnWikiSmall IndexerOutput");
    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();

        // CHECKING ARGUMENTS CORRECTNESS
        if (args.length != 2) {
            System.out.println("The number of arguments provided is incorrect");
            System.out.println("---------------------------------");
            args_usage();
            System.exit(1);
        }

        if (args[1].equals("output_idf")) {
            System.out.println("Invalid folder name. You can call your output folder by any other name");
            System.out.println("---------------------------------");
            args_usage();
            System.exit(1);
        }

        FileSystem fs = FileSystem.get(conf);
        Path p1 = new Path(args[0]);
        Path p2 = new Path(args[1]);

        if (!fs.exists(p1)) {
            System.out.println("The input directory doesn't exist");
            System.out.println("---------------------------------");
            args_usage();
            System.exit(1);
        }

        if (fs.exists(p2)) {
            System.out.println("The output folder has to be the one that does not exist yet");
            System.out.println("---------------------------------");
            args_usage();
            System.exit(1);
        }

        ///////////// IDF /////////////
        Job job = Job.getInstance(conf, "tf-idf");
        job.setJarByClass(Indexer.class);
        job.setMapperClass(IDFMapper.class);
        job.setCombinerClass(IDFReducer.class);
        job.setReducerClass(IDFReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        Path path = new Path("output_idf_tmp");
        FileOutputFormat.setOutputPath(job, path);
        job.waitForCompletion(true);
        ///////////// IDF /////////////

        ///////////// SORT IDF /////////////
        conf = new Configuration();
        Job job1 = Job.getInstance(conf, "sort");
        job1.setJarByClass(Indexer.class);
        job1.setMapperClass(SORTIDFMapper.class);
        job1.setSortComparatorClass(SORTIDFComparator.class);
        job1.setReducerClass(SORTIDFReducer.class);
        job1.setOutputKeyClass(IntWritable.class);
        job1.setOutputValueClass(Text.class);
        FileInputFormat.setInputDirRecursive(job1, true);
        FileInputFormat.addInputPath(job1, path);
        path = new Path("output_idf");
        FileOutputFormat.setOutputPath(job1, path);
        job1.waitForCompletion(true);
        ///////////// SORT IDF /////////////


        ///////////// Word2Vec & TF-IDF /////////////
        // setting configs
        conf = new Configuration();

        // initializing job
        Job job2 = Job.getInstance(conf, "indexer");
    
        // adding files to distributed cache
        fs = FileSystem.get(conf);
        try {

            // listing filenames in the dir
            FileStatus[] fileStatuses = fs.listStatus(new Path("output_idf"));

            // going through each file
            for(FileStatus status: fileStatuses) {
                String filename = status.getPath().toString();
                if (!filename.contains("SUCCESS")) { //we are not interested in _SUCCESS file
                    // reading files
                    path = new Path("output_idf/" +
                            filename.substring(filename.indexOf("output_idf/") + "output_idf/".length()));
                    job2.addCacheFile(path.toUri());
                }

            }


        } catch (IOException e){
            e.printStackTrace();
        }


        // setting corresponding classes
        job2.setJarByClass(Indexer.class);
        job2.setMapperClass(IndMapper.class);
        job2.setReducerClass(IndReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(MapWritable.class);



        // Files
        FileInputFormat.setInputDirRecursive(job2, true);
        FileInputFormat.addInputPath(job2, new Path(args[0]));
        FileOutputFormat.setOutputPath(job2, new Path(args[1]));

        // Starting the job
        job2.waitForCompletion(true);

        ///////////// Word2Vec & TF-IDF /////////////



        ///////////// AVG DOCs LEN /////////////
        int n_docs = 0; // # of documents
        int sum = 0; // sum of lens of all docs
        float avg = 0; // avg len of docs
        BufferedReader reader;
        try {

            // listing filenames in the dir
            FileStatus[] fileStatuses = fs.listStatus(new Path(args[1]));

            // going through each file
            for(FileStatus status: fileStatuses) {
                String filename = status.getPath().toString();
                if (!filename.contains("SUCCESS")) {
                    // reading files
                    path = new Path(args[1] + "/" + filename.substring(filename.indexOf(args[1]) + args[1].length() + 1));

                    reader = new BufferedReader(new InputStreamReader(fs.open(path)));

                    String line = reader.readLine();
                    while(line != null){
                        if(line.length() > 0) {
                            n_docs ++; // incrementing # of docs
                            // getting len of the doc
                            sum += Integer.parseInt(line.substring(line.indexOf("length: ") + "length: ".length(), line.indexOf("{")).replaceAll("[^0-9]", ""));
                        }
                        line = reader.readLine(); // reading the next line
                    }
                }
            }
            // calculating the avg len
            avg = (float) sum / (float) n_docs;

            FSDataOutputStream out = fs.create(new Path(args[1] + "/avg_len"));
            out.writeChars(Float.toString(avg));

        } catch (IOException e){
            e.printStackTrace();
        }
        ///////////// AVG DOCs LEN  /////////////

        System.exit(1);

    }
}

