package jenaDataGet;

import org.apache.jena.ontology.OntModelSpec;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;

public class jena_owl_get {

	public static void main(String[] args) {
		String path = "/home/jw/Desktop/GeoLink/GeoLinkIM/ontology/test.owl";
		
		Model model = ModelFactory.createOntologyModel(OntModelSpec.OWL_DL_MEM);
		model.read(path);
		model.write(System.out);

	}

}
