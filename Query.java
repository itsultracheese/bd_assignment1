import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
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


        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            // get configuration
            Configuration conf = context.getConfiguration();

            String line = value.toString(); // acquiring the doc
            // if line contains info about any doc
            if (line.contains("length")) {
                // obtain doc id
                String id = line.substring(0, line.indexOf("length")).replaceAll(" ", "");
                // obtain doc tf/idf representation
                String tfidf_t = line.substring(line.indexOf("{") + 1, line.indexOf("}"));
                line = null;    //free memory
                String cur_word;
                String cur_val;
                Float r = (float) 0;

                String[] tfidfs = tfidf_t.split(",");
                tfidf_t = null;
                // iterating through all words present in doc
                for (String t : tfidfs) {
                    cur_word = t.substring(0, t.indexOf("=")).replaceAll(" ", "");  // obtaining word
                    cur_val = t.substring(t.indexOf("=") + 1).replaceAll(" ", "");  // obtaining  word tf/idf score for doc
                    // calculate relevance score
                    r += Float.parseFloat(conf.get("!!query!!" + cur_word, "0")) * Float.valueOf(cur_val);
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

    public static HashMap<Integer, String> getTitle(Set<Integer> ids, FileSystem fs) {
        HashMap<Integer, String> result = new HashMap<Integer, String>();
        HashSet<Integer> used = new HashSet<Integer>();
        BufferedReader reader;
        Path path;
        try {

            // listing filenames in the dir
            FileStatus[] fileStatuses = fs.listStatus(new Path("/EnWikiSmall"));

            // going through each file
            for(FileStatus status: fileStatuses) {
                String filename = status.getPath().toString();

                // reading files
                path = new Path("/EnWikiSmall/" + filename.substring(filename.indexOf("/EnWikiSmall/") + "/EnWikiSmall/".length()));

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

    public static void args_usage() {
        // example of proper usage
        System.out.println("Arguments usage:");
        System.out.println("hadoop jar Query.java Query arg0 arg1 arg2 arg3");
        System.out.println("arg0 - query text in quotes");
        System.out.println("arg1 - number of relevant results to obtain, from 0 to 1000");
        System.out.println("arg2 - path to indexer step output");
        System.out.println("arg3 - path to output folder (should not exist before execution");
        System.out.println("---------------------------------");
        System.out.println("Example: hadoop jar Query.java Query \"penguin\" 10 IndexerOutput QueryOutput");
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
        if (args.length != 4) {
            System.out.println("The number of arguments provided is incorrect");
            System.out.println("---------------------------------");
            args_usage();
            System.exit(1);
        }

        Path p1 = new Path(args[2]);
        Path p2 = new Path(args[3]);

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


        ///////////// reading IDF from file /////////////
        BufferedReader reader;
        Path path;
        try {

            // listing filenames in the dir
            FileStatus[] fileStatuses = fs.listStatus(new Path("output_idf"));

            // going through each file
            for(FileStatus status: fileStatuses) {
                String filename = status.getPath().toString();
                if (!filename.contains("SUCCESS")) {
                    // reading files
                    path = new Path("output_idf/" + filename.substring(filename.indexOf("output_idf/") + "output_idf/".length()));

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
                }
            }
        } catch (IOException e){
            e.printStackTrace();
        }
        ///////////// reading IDF from file /////////////


        ///////////// QUERY TO TF-IDF /////////////
        String query = args[0];
        StringTokenizer itr = new StringTokenizer(query.toLowerCase().replaceAll("\\\\[a-z]", " ").replaceAll("-", " "));
        String word = "";
        Integer sum = 0;
        Integer cur_idf = 0;
        Float r = new Float(0); // for storing tf-idf
        HashMap<String, Integer> queryMap = new HashMap<String, Integer>(); // for storing word -> tf

        // iterating through query tokens
        while(itr.hasMoreTokens()) {
            // replacing all non-letter characters
            word = itr.nextToken().replaceAll("[\\\\0-9~`!@#$%^&*()\\-_+=\\,.<>?/'\":;{}\\[\\]\\|]", "");
            // changing tf score of the word
            if (!queryMap.containsKey(word)) {
                queryMap.put(word, 1);
            }
            else {
                queryMap.put(word, queryMap.get(word) + 1);
            }
        }

        // iterating through all words to create tf-idf score for each
        Set<String> keySet = queryMap.keySet();
        for (String k: keySet) {
            // obtaining idf for word
            cur_idf = Integer.parseInt(conf.get(k, "0"));

            if (cur_idf != 0) {
                // divide tf by idf
                r = (float) queryMap.get(k) / (float) cur_idf;

                // write query tf/idf for word for mapreduce
                conf.set("!!query!!" + String.valueOf(k.hashCode()), Float.toString(r));
            }
        }
        ///////////// QUERY TO TF-IDF /////////////



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
        FileInputFormat.addInputPath(job, new Path(args[2]));
        FileOutputFormat.setOutputPath(job, new Path(args[3]));
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
            FileStatus[] fileStatuses = fs.listStatus(new Path(args[3]));

            // going through each file
            for(FileStatus status: fileStatuses) {
                if (count >= N){
                    break;
                }

                String filename = status.getPath().toString();
                if (!filename.contains("SUCCESS")) {
                    // reading files
                    path = new Path(args[3] + "/" + filename.substring(filename.indexOf(args[3]) + args[3].length() + 1));

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
        HashMap<Integer, String> title = getTitle(top.keySet(), fs);

        // output info about top N relevant results
        for(Integer i: arr){
            System.out.println("Id: " + i.toString() + "    " + title.get(i) + "   Score: " + top.get(i).toString());
        }

        System.exit(1);
        ///////////// OBTAIN TOP-N RESULTS /////////////

    }
}
