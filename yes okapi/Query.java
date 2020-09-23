import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Query {

    public static class QueryMapper
            extends Mapper<Object, Text, FloatWritable, Text>{

        private static Float get_score(Float idf, Integer f, Integer len, Float avg_len){ //calculating okapi score
            float k1 = (float) 2;
            float b = (float) 0.75;

            return idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * len / avg_len)));
        }

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            // get configuration
            Configuration conf = context.getConfiguration();

            String line = value.toString(); // acquiring the doc
            // if line contains info about any doc
            if (line.contains("length")) {
                // obtain doc id
                String id = line.substring(0, line.indexOf("length")).replaceAll(" ", "");
                Integer len = Integer.parseInt(line.substring(line.indexOf("length:") + "length: ".length(),
                        line.indexOf("{")).replaceAll("[^0-9]", ""));

                // obtain doc tf representation
                String tf_t = line.substring(line.indexOf("{") + 1, line.indexOf("}"));
                line = null;    //free memory
                String cur_word;
                Integer cur_tf;
                Float r = (float) 0;

                String[] tfs = tf_t.split(",");
                tf_t = null;
                Float avg_len = Float.parseFloat(conf.get("avg_len", "-2"));
                // iterating through all words present in doc
                for (String t : tfs) {
                    if(t.contains("=")) {
                        cur_word = t.substring(0, t.indexOf("=")).replaceAll(" ", "");  // obtaining word
                        cur_tf = Integer.parseInt(t.substring(t.indexOf("=") + 1).replaceAll("[^0-9]", ""));  // obtaining  word tf in doc
                        // calculate relevance score
                        r += get_score(Float.parseFloat(conf.get("!!query_idf!!" + cur_word, "0")), cur_tf, len, avg_len);
                    }


                }

                context.write(new FloatWritable(r), new Text(id));
            }
        }
    }

    public static class QueryComparator
            extends WritableComparator {

        protected QueryComparator() {
            super(FloatWritable.class, true);
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            // custom comparator to sort keys (relevance score) in a descending order instead of ascending
            FloatWritable f1 = (FloatWritable)a;
            FloatWritable f2 = (FloatWritable)b;
            return f2.compareTo(f1);
        }
    }

    public static class QueryReducer
            extends Reducer<FloatWritable, Text, FloatWritable, Text> {
        private IntWritable result = new IntWritable();

        public void reduce(FloatWritable key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {

            // write sorted relevance score - doc id
            for (Text v: values) {
                context.write(key, v);
            }
        }
    }

    public static HashMap<Integer, String> getTitle(String wiki, Set<Integer> ids, FileSystem fs) {
        HashMap<Integer, String> result = new HashMap<Integer, String>();
        HashSet<Integer> used = new HashSet<Integer>();
        BufferedReader reader;
        Path path;
        try {

            // listing filenames in the dir
            FileStatus[] fileStatuses = fs.listStatus(new Path(wiki));

            // going through each file
            for(FileStatus status: fileStatuses) {
                String filename = status.getPath().toString();

                // reading files
                path = new Path(wiki + "/" + filename.substring(filename.indexOf(wiki + "/") + (wiki + "/").length()));

                reader = new BufferedReader(new InputStreamReader(fs.open(path)));

                String line = reader.readLine();
                while(line != null && used.size() != ids.size()){
                    Integer cur_id = Integer.parseInt(line.substring(8, line.indexOf("\", \"url\"")).replaceAll(" ", "")); // getting doc id
                    if(ids.contains(cur_id)) {
                        String title = line.substring(line.indexOf("title") + "title".length() + 4, line.indexOf("\", \"text"));
                        String url = line.substring(line.indexOf("url") + "url".length() + 4, line.indexOf("\", \"title"));

                        result.put(cur_id, "Title: " + title + "    URL: " + url);
                        used.add(cur_id);
                    }
                    line = reader.readLine(); // reading the next line
                }

            }
        } catch (IOException e){
            e.printStackTrace();
        }
        return result;
    }

    public static Float calc_idf(Integer n_docs, Integer docsww){
        return (float) Math.log((n_docs - docsww + 0.5) / (docsww + 0.5) + 1);
    }

    public static void args_usage() {
        // example of proper usage
        System.out.println("Arguments usage:");
        System.out.println("hadoop jar Query.jar Query arg0 arg1 arg2 arg3");
        System.out.println("arg0 - query text in quotes");
        System.out.println("arg1 - number of relevant results to obtain, from 0 to 1000");
        System.out.println("arg2 - path to indexer step output");
        System.out.println("arg3 - path to output folder (should not exist before execution");
        System.out.println("---------------------------------");
        System.out.println("Example: hadoop jar Query.jar Query \"penguin\" 10 IndexerOutput QueryOutput");
    }

    public static void main(String[] args) throws Exception {

        // arguments
        // 0 - query
        // 1 - # of relevant results
        // 2 - path to indexer output
        // 3 - output path

        // configuration and obtaining filesystem
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        // CHECKING ARGUMENTS CORRECTNESS
        if (args.length != 5) {
            System.out.println("The number of arguments provided is incorrect");
            System.out.println("---------------------------------");
            args_usage();
            System.exit(1);
        }

        Path p1 = new Path(args[3]);
        Path p2 = new Path(args[4]);
        String wiki = args[2];

        if (wiki.charAt(wiki.length() - 1) == '/') {
            wiki = wiki.substring(0, wiki.length()-1);
        }

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

        // obtaining number of relevant results desired
        Integer N = Integer.parseInt(args[1]);
        if (N < 0 || N > 1000) {
            System.out.println("The relevant results number has to be in range [0; 1000]");
            System.out.println("---------------------------------");
            args_usage();
            System.exit(1);
        }


        ///////////// reading DocsWW from file /////////////
        BufferedReader reader;
        Path path;
        HashMap<String, Integer> docsww = new HashMap<String, Integer>();
        try {

            // listing filenames in the dir
            FileStatus[] fileStatuses = fs.listStatus(new Path("output_docsww"));

            // going through each file
            for(FileStatus status: fileStatuses) {
                String filename = status.getPath().toString();
                if (!filename.contains("SUCCESS")) {
                    // reading files
                    path = new Path("output_docsww/" + filename.substring(filename.indexOf("output_docsww/") + "output_docsww/".length()));

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
                                docsww.put(cur_word, cur_idf); //saving <word, idf> for later usage
                            }
                        }

                        line = reader.readLine(); // reading the next line
                    }
                }
            }
        } catch (IOException e){
            e.printStackTrace();
        }

        //////////// obtaining avg len of docs and total # of docs /////////////
        Float avg_len = (float) -1;
        Integer n_docs = -1;
        try {

            // listing filenames in the dir
            reader = new BufferedReader(new InputStreamReader(fs.open(new Path(args[3] + "/avg_len"))));
            String line = reader.readLine();
            avg_len = Float.parseFloat(line.split(" ", 2)[0].replaceAll("[^0-9.]", ""));
            n_docs = Integer.parseInt(line.split(" ", 2)[1].replaceAll("[^0-9]", ""));

        } catch (IOException e){
            e.printStackTrace();
        }


        ///////////// QUERY TO IDF /////////////
        String query = args[0];
        StringTokenizer itr = new StringTokenizer(query.toLowerCase().replaceAll("\\\\[a-z]", " ").replaceAll("-", " "));
        String word = "";
        Integer sum = 0;
        Integer cur_docsww = 0;
        Float r = new Float(0); // for storing tf-idf
        HashSet<String> querySet = new HashSet<String>();

        // iterating through query tokens
        while(itr.hasMoreTokens()) {
            // replacing all non-letter characters
            word = itr.nextToken().replaceAll("[\\\\0-9~`!@#$%^&*()\\-_+=\\,.<>?/'\":;{}\\[\\]\\|]", "");
            querySet.add(word);
        }

        // iterating through all words to create idf score for each
        for (String k: querySet) {
            // obtaining docsww for word
            cur_docsww = docsww.get(k);

            if (cur_docsww != null) {
                r = calc_idf(n_docs, cur_docsww);

                System.out.println("r = " + r);
                // write query idf for word for mapreduce
                conf.set("!!query_idf!!" + String.valueOf(k.hashCode()), Float.toString(r));
            }

        }

        conf.set("avg_len", avg_len.toString());



        // mapreduce job
        Job job = Job.getInstance(conf, "query");


        job.setJarByClass(Query.class);
        job.setMapperClass(QueryMapper.class);
        job.setCombinerClass(QueryReducer.class);
        job.setSortComparatorClass(QueryComparator.class);
        job.setReducerClass(QueryReducer.class);
        job.setOutputKeyClass(FloatWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[3]));
        FileOutputFormat.setOutputPath(job, new Path(args[4]));
        job.waitForCompletion(true);


        ///////////// OBTAIN TOP-N RESULTS /////////////
        Integer count = 0;
        Float cur_score = (float) 0;
        Integer cur_id = 0;
        // doc id - relevancy score
        HashMap<Integer, Float> top = new HashMap<Integer, Float>();
        // top doc ids
        ArrayList<Integer> arr = new ArrayList<Integer>();

        try {

            // listing filenames in the dir
            FileStatus[] fileStatuses = fs.listStatus(new Path(args[4]));

            // going through each file
            for(FileStatus status: fileStatuses) {
                if (count >= N){
                    break;
                }

                String filename = status.getPath().toString();
                if (!filename.contains("SUCCESS")) {
                    // reading files
                    path = new Path(args[4] + "/" + filename.substring(filename.indexOf(args[4]) + args[4].length() + 1));

                    reader = new BufferedReader(new InputStreamReader(fs.open(path)));

                    String line = reader.readLine();
                    while(line != null && count < Integer.parseInt(args[1])){
                        itr = new StringTokenizer(line);
                        // iterating through line
                        if(itr.hasMoreTokens()){
                            // reading relevancy score value
                            cur_score = Float.parseFloat(itr.nextToken());
                            if (itr.hasMoreTokens()) {
                                // reading doc id
                                cur_id = Integer.parseInt(itr.nextToken().replaceAll("[^0-9]", ""));
                                // saving doc id and score for later output
                                top.put(new Integer(cur_id), new Float(cur_score));
                                arr.add(cur_id);
                            }
                        }

                        count++;
                        line = reader.readLine(); // reading the next line
                    }
                }
            }
        } catch (IOException e){
            e.printStackTrace();
        }

        // obtaining titles and urls
        HashMap<Integer, String> title = getTitle(wiki, top.keySet(), fs);

        // output info about top N relevant results
        for(Integer i: arr){
            System.out.println("Id: " + i.toString() + "    " + title.get(i) + "   Score: " + top.get(i).toString());
        }

        System.exit(1);

    }
}
