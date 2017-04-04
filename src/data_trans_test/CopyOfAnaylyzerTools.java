package data_trans_test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.lionsoul.jcseg.ASegment;
import org.lionsoul.jcseg.core.ADictionary;
import org.lionsoul.jcseg.core.DictionaryFactory;
import org.lionsoul.jcseg.core.IWord;
import org.lionsoul.jcseg.core.JcsegException;

import org.lionsoul.jcseg.core.JcsegTaskConfig;
import org.lionsoul.jcseg.core.SegmentFactory;

public class CopyOfAnaylyzerTools {
	public JcsegTaskConfig config = new JcsegTaskConfig(
			CopyOfAnaylyzerTools.class.getResource("").getPath()
					+ "jcseg.properties");
	public ADictionary dic = DictionaryFactory.createDefaultDictionary(config);

	public ArrayList<String> anaylyzerWords(String str, JcsegTaskConfig config,
			ADictionary dic) {

		ArrayList<String> list = new ArrayList<String>();
		ASegment seg = null;
		try {
			seg = (ASegment) SegmentFactory.createJcseg(
					JcsegTaskConfig.COMPLEX_MODE, new Object[] { config, dic });

		} catch (JcsegException e1) {
			e1.printStackTrace();
		}
		try {
			seg.reset(new StringReader(str));
			IWord word = null;
			while ((word = seg.next()) != null) {
				String spwords = word.getValue();
				// 只过滤出了中文 字母要不要？
				String result = spwords.replaceAll("[^(\\u4e00-\\u9fa5)]", "");
				if (result.length() > 1)
					list.add(result);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.runFinalization();
		return list;
	}

	// 测试分词
	public static void main(String[] args) {
		long start = System.nanoTime();
		String path = "C:/Users/Administrator/Desktop/abs.txt";
		// String path = "C:/Users/Administrator/Desktop/ouput/savechar"; // 源文件路径
		// 输出文件路径
		String resPath = "C:/Users/Administrator/Desktop/splitoutput/";
		int i = 0; // 读取的行数标记，每轮读取100行
		BufferedReader br = null;
		String str = null;
		try {
			br = new BufferedReader(new FileReader(path));
			CopyOfAnaylyzerTools anaylyzerTools = new CopyOfAnaylyzerTools();
			JcsegTaskConfig configg = anaylyzerTools.config;
			ADictionary dicc = anaylyzerTools.dic;
			while ((str = br.readLine()) != null) {
				i++;
				// StringReader reader = new StringReader(str);
				List<String> list = anaylyzerTools.anaylyzerWords(str, configg,
						dicc);

				String splitstr = list.toString().replace("[", "")
						.replace("]", "").replaceAll(" ", "")
						.replaceAll(",", " ").replaceAll("[(]", "")
						.replaceAll("[)]", "");
				// 文件落地 文件名是对应的索引号
				FileWriter fr = new FileWriter(resPath + i + ".txt");
				fr.write(splitstr);
				System.out.println();
				fr.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		long end = System.nanoTime();
		double diff = (end - start) / 1e9;
		System.out.println("time: " + diff);

	}

	// 调用分词
	public static void split() {
		long start = System.nanoTime();
		String path = "C:/Users/Administrator/Desktop/ouput/savechar"; // 源文件路径
		// 输出文件路径
		String resPath = "C:/Users/Administrator/Desktop/splitoutput/";
		int i = 0; // 读取的行数标记，每轮读取100行
		BufferedReader br = null;
		String str = null;
		try {
			br = new BufferedReader(new FileReader(path));
			CopyOfAnaylyzerTools anaylyzerTools = new CopyOfAnaylyzerTools();
			JcsegTaskConfig configg = anaylyzerTools.config;
			ADictionary dicc = anaylyzerTools.dic;
			while ((str = br.readLine()) != null) {
				i++;
				// StringReader reader = new StringReader(str);
				List<String> list = anaylyzerTools.anaylyzerWords(str, configg,
						dicc);
				String splitstr = list.toString().replace("[", "")
						.replace("]", "").replaceAll(" ", "")
						.replaceAll(",", " ").replaceAll("[(]", "")
						.replaceAll("[)]", "");
				FileWriter fr = new FileWriter(resPath + i);
				fr.write(splitstr);
				System.out.println();
				fr.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		long end = System.nanoTime();
		double diff = (end - start) / 1e9;
		System.out.println("time: " + diff);

	}
}
