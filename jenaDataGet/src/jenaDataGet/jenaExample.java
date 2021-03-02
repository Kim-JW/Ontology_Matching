package jenaDataGet;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.HashSet;

import org.apache.jena.rdf.*;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.rdf.model.StmtIterator;

public class jenaExample {

	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
		
		FileInputStream fis = new FileInputStream("src/jenaDataGet/Dataset/ekaw.owl");
		InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
		Model model = ModelFactory.createDefaultModel();
		model.read(isr, "", "RDF/XML");
		StringBuilder stringBuilder = new StringBuilder();
		StmtIterator iter = model.listStatements();
		HashSet<Statement> result = new HashSet<>();
		while (iter.hasNext()) {
			Statement stmt = iter.nextStatement(); // get next statement
			// write in RDFHashCode.txt
			/*
			stringBuilder.append("<"+stmt.getSubject().toString() + "><" + stmt.getPredicate().toString() + "><"
					+ stmt.getObject().toString() + ">\r\n");
			result.add(stmt);
			*/
			System.out.println(stmt.getSubject() +","+ stmt.getPredicate());
		}
		
	}
}

// /home/jw/Desktop/conference/conference/ekaw.owl
