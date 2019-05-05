package data_trans_test;

import java.io.File;
import java.io.IOException;

public class SaveFile {
	public static void makeDir(String str) {
		File file = new File(str); 
		//创建文件目录
		file.mkdirs();
	}

}
