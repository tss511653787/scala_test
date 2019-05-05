package utils;

import org.lionsoul.jcseg.tokenizer.ASegment;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import org.lionsoul.jcseg.tokenizer.core.ADictionary;
import org.lionsoul.jcseg.tokenizer.core.DictionaryFactory;
import org.lionsoul.jcseg.tokenizer.core.IWord;
import org.lionsoul.jcseg.tokenizer.core.JcsegException;
import org.lionsoul.jcseg.tokenizer.core.JcsegTaskConfig;
import org.lionsoul.jcseg.tokenizer.core.SegmentFactory;

public class AnaylyzerTools {
	// load properties & Dictionary
	JcsegTaskConfig config = new JcsegTaskConfig(AnaylyzerTools.class
			.getResource("").getPath() + "jcseg.properties");
	ADictionary dic = DictionaryFactory.createDefaultDictionary(config);

	public ArrayList<String> anaylyzerWords(String str) {
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
		// String str =
		// "阿埃吉尔田申申：40万以内的SUV，哪个性价比高？,GLC   X3  SRX  发现神行  Q5    哪个好？？？不需要越野，城市开开……沃尔沃XC60,40万以内的SUV，哪个性价比高？ GLC   X3  SRX  发现神行  Q5    哪个好？？？不需要越野，城市开开…… 沃尔沃XC60";
		String str = "《血战钢锯岭》是熙颐影业出品的战争历史片，由梅尔•吉布森执导，影片改编自二战上等兵军医戴斯蒙德•道斯的真实经历，讲述他拒绝携带武器上战场，并在冲绳战役中赤手空拳救下75位战友的传奇故事。影片于2016年12月08日在中国上映。2016年12月，《血战钢锯岭》被选为2016美国电影学会十佳电影。";
		// List<String> list = AnaylyzerTools.anaylyzerWords(str);
		AnaylyzerTools anaylyzerTools = new AnaylyzerTools();
		List<String> list = anaylyzerTools.anaylyzerWords(str);
		System.out.println(str);
		for (String word : list) {
			System.out.print(word + "/");
		}
		System.out.println();
		System.out.println(list.size());
	}
}
