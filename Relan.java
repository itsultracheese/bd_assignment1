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
import org.apache.jute.Index;

public class Relan {

    public static class RelanMapper
            extends Mapper<Object, Text, FloatWritable, Text>{


        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();

            String line = value.toString(); // acquiring the doc
            if (line.contains("length")) {
                String id = line.substring(0, line.indexOf("length")).replaceAll(" ", "");
                String tfidf_t = line.substring(line.indexOf("{") + 1, line.indexOf("}"));
                line = null;
                String cur_word;
                String cur_val;
                Float r = (float) 0;

                String[] tfidfs = tfidf_t.split(",");
                tfidf_t = null;
                for (String t : tfidfs) {
                    cur_word = t.substring(0, t.indexOf("=")).replaceAll(" ", "");
                    cur_val = t.substring(t.indexOf("=") + 1).replaceAll(" ", "");
                    r += Float.parseFloat(conf.get("!!query!!" + cur_word, "0")) * Float.valueOf(cur_val);
                }

                context.write(new FloatWritable(r), new Text(id));
            }
        }
    }

    public static class RelanComparator
            extends WritableComparator {

        protected RelanComparator() {
            super(FloatWritable.class, true);
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            FloatWritable f1 = (FloatWritable)a;
            FloatWritable f2 = (FloatWritable)b;
            return f2.compareTo(f1);
        }
    }

    public static class RelanReducer
            extends Reducer<FloatWritable, Text, FloatWritable, Text> {
        private IntWritable result = new IntWritable();

        public void reduce(FloatWritable key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {

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

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        // 0 - query
        // 1 - # of relevant results
        // 2 - path to indexer output
        // 3 - output path

        ///////////// reading IDF from file /////////////
        FileSystem fs = FileSystem.get(conf);
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
        Float r = new Float(0);
        HashMap<String, Integer> queryMap = new HashMap<String, Integer>();


        while(itr.hasMoreTokens()) {
            word = itr.nextToken().replaceAll("[\\\\0-9~`!@#$%^&*()\\-_+=\\,.<>?/'\":;{}\\[\\]\\|]", "");
            if (!queryMap.containsKey(word)) {
                queryMap.put(word, 1);
            }
            else {
                queryMap.put(word, queryMap.get(word) + 1);
            }
        }

        Set<String> keySet = queryMap.keySet();
        for (String k: keySet) {
            cur_idf = Integer.parseInt(conf.get(k, "0"));
            r = (float) queryMap.get(k) / (float) cur_idf;

            conf.set("!!query!!" + String.valueOf(k.hashCode()), Float.toString(r));
        }
        ///////////// QUERY TO TF-IDF /////////////


        Job job = Job.getInstance(conf, "relan");
        job.setJarByClass(Relan.class);
        job.setMapperClass(RelanMapper.class);
        job.setCombinerClass(RelanReducer.class);
        job.setSortComparatorClass(RelanComparator.class);
        job.setReducerClass(RelanReducer.class);
        job.setOutputKeyClass(FloatWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[2]));
        FileOutputFormat.setOutputPath(job, new Path(args[3]));
        job.waitForCompletion(true);

        Integer count = 0;
        Float cur_score = (float) 0;
        Integer cur_id = 0;
        HashMap<Integer, Float> top = new HashMap<Integer, Float>();
        ArrayList<Integer> arr = new ArrayList<Integer>();

        try {

            // listing filenames in the dir
            FileStatus[] fileStatuses = fs.listStatus(new Path(args[3]));

            // going through each file
            for(FileStatus status: fileStatuses) {
                if (count >= Integer.parseInt(args[1])){
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
                            cur_score = Float.parseFloat(itr.nextToken());
//                            System.out.println(cur_score);
                            if (itr.hasMoreTokens()) {
                                cur_id = Integer.parseInt(itr.nextToken().replaceAll("[^0-9]", ""));
                                top.put(new Integer(cur_id), new Float(cur_score));
//                                System.out.println(top.get(cur_id));
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

        HashMap<Integer, String> title = getTitle(top.keySet(), fs);

        for(Integer i: arr){
//            System.out.println(top.keySet());
            System.out.println("Id: " + i.toString() + "    " + title.get(i) + "   Score: " + top.get(i).toString());
        }

        System.exit(1);

    }
}
