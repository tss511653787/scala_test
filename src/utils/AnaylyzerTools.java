package utils;

import org.lionsoul.jcseg.ASegment;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import org.lionsoul.jcseg.core.ADictionary;
import org.lionsoul.jcseg.core.DictionaryFactory;
import org.lionsoul.jcseg.core.IWord;
import org.lionsoul.jcseg.core.JcsegException;
import org.lionsoul.jcseg.core.JcsegTaskConfig;
import org.lionsoul.jcseg.core.SegmentFactory;

public class AnaylyzerTools {
	public static ArrayList<String> anaylyzerWords(String str) {
		JcsegTaskConfig config = new JcsegTaskConfig(AnaylyzerTools.class
				.getResource("").getPath() + "jcseg.properties");
		ADictionary dic = DictionaryFactory.createDefaultDictionary(config);
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
				list.add(word.getValue());
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.runFinalization();
		return list;
	}

	// 测试分词
	public static void main(String[] args) {
		String str = "阿埃吉尔田申申：40万以内的SUV，哪个性价比高？,GLC   X3  SRX  发现神行  Q5    哪个好？？？不需要越野，城市开开……沃尔沃XC60,40万以内的SUV，哪个性价比高？ GLC   X3  SRX  发现神行  Q5    哪个好？？？不需要越野，城市开开…… 沃尔沃XC60";
		// List<String> list = AnaylyzerTools.anaylyzerWords(str);
		AnaylyzerTools anaylyzerTools = new AnaylyzerTools();
		List<String> list = anaylyzerTools.anaylyzerWords(str);
		System.out.println(str);
		for (String word : list) {
			System.out.print(word + ",");
		}
		System.out.println();
		System.out.println(list.size());
	}
}
