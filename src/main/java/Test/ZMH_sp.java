package Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import utils.AnaylyzerTools;

public class ZMH_sp {
	static Scanner scanner;
	static String file_path = "C://Users//Administrator//Desktop//sp//segment_data.txt";
	static String outpath = "C://Users//Administrator//Desktop//sp//out.txt";
	static File file_in;
	static File file_out;
	static List<String> list;
	static PrintWriter out;

	public static void main(String[] args) throws FileNotFoundException {
		long start = System.nanoTime();
		file_in = new File(file_path);
		scanner = new Scanner(file_in);
		list = new ArrayList<String>();
		String in = scanner.nextLine();

		AnaylyzerTools anaylyzerTools = new AnaylyzerTools();
		list = anaylyzerTools.anaylyzerWords(in);
		file_out = new File(outpath);
		out = new PrintWriter(file_out);
		for (String s : list) {
			out.print(s);
			out.println();
		}
		out.close();
		long end = System.nanoTime();
		System.out.println((end - start) / 1e9);
	}

}
