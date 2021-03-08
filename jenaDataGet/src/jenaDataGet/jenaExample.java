package jenaDataGet;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashSet;

import org.apache.jena.rdf.*;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.rdf.model.StmtIterator;

public class jenaExample {

	public static void main(String[] args) throws IOException {
		
		FileInputStream fis = new FileInputStream("src/jenaDataGet/Dataset/gbo/gbobcodmo.owl");
		InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
		
		FileOutputStream fo = new FileOutputStream("gmobcodmo.txt");
		FileWriter fw = new FileWriter("gbor2r2RDF_new.txt",true);
		
		Model model = ModelFactory.createDefaultModel();
		model.read(isr, "", "TURTLE");
		
		StringBuilder stringBuilder = new StringBuilder();
		
		StmtIterator iter = model.listStatements();
		
		HashSet<Statement> result = new HashSet<>();
		
		int cnt = 0;
		
		while (iter.hasNext()) {
			
			Statement stmt = iter.nextStatement(); // get next statement
			// write in RDFHashCode.txt
			/*
			stringBuilder.append("<"+stmt.getSubject().toString() + "><" + stmt.getPredicate().toString() + "><"
					+ stmt.getObject().toString() + ">\r\n");
			result.add(stmt);
			*/
			
			stringBuilder.append(stmt.getSubject().toString() + "@" + stmt.getPredicate().toString() + "@" + stmt.getObject().toString() + "\n");
			
		}
		
		fw.write(stringBuilder.toString());
		
		fw.close();
		
	}
}

// /home/jw/Desktop/conference/conference/ekaw.owl
